"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口

增加熔断保护和 Embedding 缓存，失败时返回 None 而非抛异常，
让调用方可以路由到 BM25-only 降级路径。
"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config
from app.core.circuit_breaker import get_breaker, BREAKER_EMBEDDING, CircuitOpenError
from app.core.health_registry import health_registry
from app.core.cache import embedding_cache, make_cache_key


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding (OpenAI 兼容模式)

    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimensions

        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档 (LangChain 标准接口)"""
        if not texts:
            return []

        breaker = get_breaker(BREAKER_EMBEDDING)
        try:
            breaker.before_call()
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"批量嵌入完成, 维度: {len(embeddings[0])}")
            breaker.record_success()
            health_registry.mark_success("dashscope_embedding")
            return embeddings

        except CircuitOpenError:
            logger.error("Embedding API 熔断，批量嵌入不可用")
            raise RuntimeError("Embedding API 熔断") from None
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("dashscope_embedding")
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本 (LangChain 标准接口)"""
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        # 先查缓存
        cache_key = make_cache_key("emb", text)
        cached = embedding_cache.get(cache_key)
        if cached is not None:
            logger.debug("Embedding 命中缓存")
            return cached

        breaker = get_breaker(BREAKER_EMBEDDING)
        try:
            breaker.before_call()
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )

            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            breaker.record_success()
            health_registry.mark_success("dashscope_embedding")

            # 写入缓存
            embedding_cache.set(cache_key, embedding)
            return embedding

        except CircuitOpenError:
            logger.error("Embedding API 熔断，查询嵌入不可用")
            raise RuntimeError("Embedding API 熔断") from None
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("dashscope_embedding")
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e

    def embed_query_safe(self, text: str) -> Optional[List[float]]:
        """安全版本：失败返回 None 而非抛异常，供降级路径使用"""
        try:
            return self.embed_query(text)
        except Exception as e:
            logger.warning(f"Embedding 安全调用失败: {e}，将走 BM25-only 路径")
            return None


# 全局单例
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=1024
)
