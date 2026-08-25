"""LLM 工厂类

通过 OpenAI 兼容模式调用 OpenRouter（模型：NVIDIA Nemotron 3.5 Lightning）。
支持双层模型（strong/weak）+ 实例缓存 + 降级链 + 超时控制 + 熔断保护。

降级链：主模型 → 弱模型 → 缓存 → 模板响应。

限流（429）是免费档的运营常态而非故障：识别后按 Retry-After/指数退避
做有限次等待重试，且不计入熔断失败——否则几个并发请求撞限流就能把
熔断打 OPEN，形成「限流→熔断→全员模板响应」的放大链。
"""

import asyncio
import random

from langchain_openai import ChatOpenAI
from loguru import logger
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import SecretStr

from app.config import config
from app.core.cache import llm_response_cache, make_cache_key
from app.core.circuit_breaker import BREAKER_LLM, CircuitOpenError, get_breaker
from app.core.cost_tracker import cost_tracker
from app.core.degradation import DegradationLevel, get_template_response


class RateLimitExhausted(Exception):
    """LLM 限流退避重试耗尽：不计熔断失败，也不应再打同源免费档备用模型"""


def _is_rate_limit(e: BaseException) -> bool:
    """识别限流错误：openai.RateLimitError 或带 status_code=429 的异常"""
    if isinstance(e, OpenAIRateLimitError):
        return True
    return getattr(e, "status_code", None) == 429


class LLMFactory:
    """LLM 工厂类 - 双层模型 + 实例缓存 + 降级链 + 超时 + 熔断"""

    # 实例缓存：(model, temperature, streaming, base_url, api_key, timeout) → ChatOpenAI
    # 避免每个节点/每步重复构造客户端与重拉配置
    _instances: dict[tuple, ChatOpenAI] = {}

    # 429 处理参数：额外重试次数与退避上限（分钟级限流窗口等不起太久，
    # 等待超过封顶值的场景交给上层降级链，而不是把请求挂死）
    _RATE_LIMIT_RETRIES = 1
    _RATE_LIMIT_BACKOFF_CAP = 8.0

    @classmethod
    def create_chat_model(
        cls,
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout: float | None = None,
        max_retries: int = 0,
    ) -> ChatOpenAI:
        """创建（或复用缓存的）聊天模型实例

        Args:
            model: OpenRouter 模型 slug，默认使用 rag_model（Nemotron 3.5 Lightning）
            temperature: 采样温度
            streaming: 是否流式
            base_url: OpenAI 兼容端点，默认 OpenRouter
            api_key: API Key；仅当调用方显式注入时绕过实例缓存
                （测试隔离语义），生产路径一律从配置读取并复用缓存实例
            request_timeout: 单次请求超时（秒），默认 config.llm_timeout_default。
                此前未设置会落到 SDK 默认的数百秒长超时——SSE 停滞时流挂死
            max_retries: SDK 层盲重试次数，默认 0。限流等待由
                invoke_with_fallback 按 Retry-After 显式处理，SDK 盲重试
                只会放大延迟并掩盖真实失败语义
        """
        explicit_key = api_key
        model = model or config.rag_model
        base_url = base_url or config.openrouter_base_url
        api_key = explicit_key or config.openrouter_api_key
        timeout_value = (
            request_timeout if request_timeout is not None else config.llm_timeout_default
        )

        cache_key = (model, temperature, streaming, base_url, api_key, timeout_value)
        cached = cls._instances.get(cache_key)
        if cached is not None:
            return cached

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=SecretStr(api_key) if api_key else None,
            timeout=timeout_value,  # request_timeout 字段的 alias（mypy 只认 alias）
            max_retries=max_retries,
        )
        if explicit_key is None:  # 默认凭证路径进缓存；显式注入的实例不缓存、不污染
            cls._instances[cache_key] = llm
        return llm

    @classmethod
    def strong(cls, temperature: float = 0.0) -> ChatOpenAI:
        """强模型层：规划/报告等复杂任务（config.llm_timeout_complex 长超时）"""
        return cls.create_chat_model(
            model=config.rag_model,
            temperature=temperature,
            request_timeout=config.llm_timeout_complex,
        )

    @classmethod
    def weak(cls, temperature: float = 0.0) -> ChatOpenAI:
        """弱模型层：路由/改写/打分等简单任务（config.llm_timeout_simple 快、省）"""
        return cls.create_chat_model(
            model=config.llm_backup_model,
            temperature=temperature,
            request_timeout=config.llm_timeout_simple,
        )

    # ──────────────── 429 限流处理 ────────────────

    @classmethod
    def _rate_limit_delay(cls, exc: Exception, attempt: int) -> float:
        """退避间隔：优先尊重 Retry-After（封顶），否则指数退避 + 抖动"""
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            seconds: float = float(retry_after)
            return min(seconds, cls._RATE_LIMIT_BACKOFF_CAP)
        base: float = 2.0 * (2**attempt)  # 显式标注：int**int 会命中 Any 重载
        return min(base + random.uniform(0, 0.5), cls._RATE_LIMIT_BACKOFF_CAP)

    @classmethod
    async def _ainvoke_with_rate_limit_retry(
        cls, llm: ChatOpenAI, prompt: str, timeout_seconds: float
    ):
        """带限流感知的调用：429 按 Retry-After/指数退避做有限次等待重试

        - 非 429 异常原样上抛，走调用方既有的熔断/降级路径；
        - 429 重试耗尽抛 RateLimitExhausted（不计熔断、不触发同源备用）。
        """
        attempts = cls._RATE_LIMIT_RETRIES + 1
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout_seconds)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                if attempt == attempts - 1:
                    raise RateLimitExhausted(str(e)) from e
                delay = cls._rate_limit_delay(e, attempt)
                logger.warning(f"LLM 限流(429)，{delay:.1f}s 后重试 ({attempt + 1}/{attempts - 1})")
                await asyncio.sleep(delay)

    @staticmethod
    async def invoke_with_fallback(
        prompt: str,
        *,
        context: str = "KNOWLEDGE",
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> tuple[str, DegradationLevel]:
        """带降级链的 LLM 调用

        Level 0: 主模型 (nemotron-3.5-lightning)
        Level 1: 备用弱模型 (llm_backup_model)
        Level 2: 缓存
        Level 3: 模板响应

        Returns:
            (响应文本, 降级等级)
        """
        breaker = get_breaker(BREAKER_LLM)
        cache_key = make_cache_key(prompt) if use_cache else ""

        # Level 0: 主模型
        try:
            breaker.before_call()
            llm = LLMFactory.create_chat_model(
                model=config.rag_model,
                temperature=temperature,
                streaming=False,
            )
            result = await LLMFactory._ainvoke_with_rate_limit_retry(llm, prompt, timeout_seconds)
            breaker.record_success()
            content = str(result.content) if hasattr(result, "content") else str(result)

            # 成本追踪
            usage = getattr(result, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
            cost_tracker.record(
                model=config.rag_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                scene=context,
            )

            if use_cache and cache_key:
                llm_response_cache.set(cache_key, content)
            return content, DegradationLevel.NONE
        except CircuitOpenError:
            logger.warning("LLM 主模型熔断，尝试备用模型")
        except RateLimitExhausted:
            # 免费档主备模型共享同一 key/配额：主模型 429 时备用必然同样 429。
            # 不计熔断失败（限流是节奏问题不是故障）、不白付一次备用调用的
            # 失败延迟——直接落缓存/模板响应（return 跳过整个备用层）。
            logger.warning("LLM 主模型限流重试耗尽，跳过同源备用模型，进入缓存/模板")
            return await LLMFactory._cache_or_template(cache_key, context, use_cache)
        except TimeoutError:
            breaker.record_failure()
            logger.warning(f"LLM 主模型超时 ({timeout_seconds}s)，尝试备用模型")
        except Exception as e:
            breaker.record_failure()
            logger.warning(f"LLM 主模型调用失败: {e}，尝试备用模型")

        # Level 1: 备用弱模型
        try:
            backup_llm = LLMFactory.create_chat_model(
                model=config.llm_backup_model,
                temperature=temperature,
                streaming=False,
            )
            result = await LLMFactory._ainvoke_with_rate_limit_retry(
                backup_llm, prompt, timeout_seconds
            )
            content = str(result.content) if hasattr(result, "content") else str(result)

            # 成本追踪
            usage = getattr(result, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
            cost_tracker.record(
                model=config.llm_backup_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                scene=context,
            )

            if use_cache and cache_key:
                llm_response_cache.set(cache_key, content)
            logger.info("LLM 备用模型调用成功")
            return content, DegradationLevel.BACKUP_MODEL
        except Exception as e:
            logger.warning(f"LLM 备用模型也失败: {e}")

        # Level 2/3: 缓存 → 模板
        return await LLMFactory._cache_or_template(cache_key, context, use_cache)

    @staticmethod
    async def _cache_or_template(
        cache_key: str, context: str, use_cache: bool
    ) -> tuple[str, DegradationLevel]:
        """降级链收尾：缓存（Level 2）→ 模板响应（Level 3）"""
        if use_cache and cache_key:
            cached = llm_response_cache.get(cache_key)
            if cached:
                logger.info("LLM 返回缓存结果")
                return cached, DegradationLevel.CACHED

        logger.warning("LLM 全部不可用，返回模板响应")
        return get_template_response(context), DegradationLevel.TEMPLATE


# 全局 LLM 工厂实例
llm_factory = LLMFactory()
