"""LLM 工厂类

通过 OpenAI 兼容模式调用 OpenRouter（模型：NVIDIA Nemotron 3.5 Lightning）。
支持双层模型（strong/weak）+ 实例缓存 + 降级链 + 超时控制 + 熔断保护。

降级链：主模型 → 弱模型 → 缓存 → 模板响应。
"""

import asyncio

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

from app.config import config
from app.core.cache import llm_response_cache, make_cache_key
from app.core.circuit_breaker import BREAKER_LLM, CircuitOpenError, get_breaker
from app.core.cost_tracker import cost_tracker
from app.core.degradation import DegradationLevel, get_template_response


class LLMFactory:
    """LLM 工厂类 - 双层模型 + 实例缓存 + 降级链 + 超时 + 熔断"""

    # 实例缓存：(model, temperature, streaming) → ChatOpenAI
    # 避免每个节点/每步重复构造客户端与重拉配置
    _instances: dict[tuple, ChatOpenAI] = {}

    @classmethod
    def create_chat_model(
        cls,
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        """创建（或复用缓存的）聊天模型实例

        Args:
            model: OpenRouter 模型 slug，默认使用 rag_model（Nemotron 3.5 Lightning）
            temperature: 采样温度
            streaming: 是否流式
            base_url: OpenAI 兼容端点，默认 OpenRouter
            api_key: API Key；仅当调用方显式注入时绕过实例缓存
                （测试隔离语义），生产路径一律从配置读取并复用缓存实例
        """
        explicit_key = api_key
        model = model or config.rag_model
        base_url = base_url or config.openrouter_base_url
        api_key = explicit_key or config.openrouter_api_key

        cache_key = (model, temperature, streaming, base_url, api_key)
        cached = cls._instances.get(cache_key)
        if cached is not None:
            return cached

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=SecretStr(api_key) if api_key else None,
        )
        if explicit_key is None:  # 默认凭证路径进缓存；显式注入的实例不缓存、不污染
            cls._instances[cache_key] = llm
        return llm

    @classmethod
    def strong(cls, temperature: float = 0.0) -> ChatOpenAI:
        """强模型层：规划/报告等复杂任务（长超时）"""
        return cls.create_chat_model(model=config.rag_model, temperature=temperature)

    @classmethod
    def weak(cls, temperature: float = 0.0) -> ChatOpenAI:
        """弱模型层：路由/改写/打分等简单任务（快、省）"""
        return cls.create_chat_model(model=config.llm_backup_model, temperature=temperature)

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
            result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout_seconds)
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
            result = await asyncio.wait_for(backup_llm.ainvoke(prompt), timeout=timeout_seconds)
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

        # Level 2: 缓存
        if use_cache and cache_key:
            cached = llm_response_cache.get(cache_key)
            if cached:
                logger.info("LLM 返回缓存结果")
                return cached, DegradationLevel.CACHED

        # Level 3: 模板响应
        logger.warning("LLM 全部不可用，返回模板响应")
        return get_template_response(context), DegradationLevel.TEMPLATE


# 全局 LLM 工厂实例
llm_factory = LLMFactory()
