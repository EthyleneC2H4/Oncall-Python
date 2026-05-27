"""LLM 工厂类

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope。
支持降级链：qwen-max → qwen-plus → 缓存 → 模板响应。
所有调用附带超时控制和熔断保护。
"""

import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config
from app.core.circuit_breaker import get_breaker, BREAKER_LLM, CircuitOpenError
from app.core.cache import llm_response_cache, make_cache_key
from app.core.degradation import DegradationLevel, get_template_response


class LLMFactory:
    """LLM 工厂类 - 支持降级链 + 超时 + 熔断"""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 备用模型（更快、更便宜）
    BACKUP_MODEL = "qwen-plus"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key

        extra_body = {"stream": streaming}

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
        )

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

        Level 0: 主模型 (qwen-max)
        Level 1: 备用模型 (qwen-plus)
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
                model=config.dashscope_model,
                temperature=temperature,
                streaming=False,
            )
            result = await asyncio.wait_for(
                llm.ainvoke(prompt), timeout=timeout_seconds
            )
            breaker.record_success()
            content = result.content if hasattr(result, "content") else str(result)
            if use_cache and cache_key:
                llm_response_cache.set(cache_key, content)
            return content, DegradationLevel.NONE
        except CircuitOpenError:
            logger.warning("LLM 主模型熔断，尝试备用模型")
        except asyncio.TimeoutError:
            breaker.record_failure()
            logger.warning(f"LLM 主模型超时 ({timeout_seconds}s)，尝试备用模型")
        except Exception as e:
            breaker.record_failure()
            logger.warning(f"LLM 主模型调用失败: {e}，尝试备用模型")

        # Level 1: 备用模型
        try:
            backup_llm = LLMFactory.create_chat_model(
                model=LLMFactory.BACKUP_MODEL,
                temperature=temperature,
                streaming=False,
            )
            result = await asyncio.wait_for(
                backup_llm.ainvoke(prompt), timeout=timeout_seconds
            )
            content = result.content if hasattr(result, "content") else str(result)
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
