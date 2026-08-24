"""向量嵌入服务模块 - 本地 sentence-transformers 模型

使用本地 BGE 中文模型（默认 BAAI/bge-large-zh-v1.5，1024 维），
实现 LangChain Embeddings 标准接口。模型懒加载（首次调用时才载入显存/内存），
带缓存与健康探针；失败时抛异常，由调用方路由到 BM25-only 降级路径。
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from loguru import logger

from app.config import config
from app.core.cache import embedding_cache, make_cache_key
from app.core.circuit_breaker import BREAKER_EMBEDDING, CircuitOpenError, get_breaker
from app.core.health_registry import health_registry

# BGE 中文模型官方推荐的查询侧指令前缀（仅用于 query，不用于文档）
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _auto_device() -> str:
    """自动选择推理设备：mps（Apple Silicon GPU）> cpu"""
    if config.embedding_device:
        return config.embedding_device
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


class LocalEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地向量模型

    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
    ):
        self.model_name = model_name or config.embedding_model
        self.dimensions = dimensions or config.embedding_dimensions
        self._model = None  # 懒加载：首次调用时载入
        logger.info(
            f"本地 Embedding 服务初始化完成 - "
            f"模型: {self.model_name}, 维度: {self.dimensions}（模型将在首次调用时加载）"
        )

    def _ensure_loaded(self):
        """懒加载底层模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = _auto_device()
            logger.info(f"加载本地向量模型 {self.model_name} (device={device})...")
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info("本地向量模型加载完成")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档 (LangChain 标准接口)"""
        if not texts:
            return []

        breaker = get_breaker(BREAKER_EMBEDDING)
        try:
            breaker.before_call()
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            model = self._ensure_loaded()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = [e.tolist() for e in embeddings]
            logger.debug(f"批量嵌入完成, 维度: {len(vectors[0])}")
            breaker.record_success()
            health_registry.mark_success("embedding")
            return vectors

        except CircuitOpenError:
            logger.error("Embedding 熔断，批量嵌入不可用")
            raise RuntimeError("Embedding 熔断") from None
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("embedding")
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本 (LangChain 标准接口)"""
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        # 先查缓存
        cache_key = make_cache_key("emb", text)
        cached = embedding_cache.get(cache_key)
        if isinstance(cached, list):
            logger.debug("Embedding 命中缓存")
            return [float(x) for x in cached]

        breaker = get_breaker(BREAKER_EMBEDDING)
        try:
            breaker.before_call()
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            model = self._ensure_loaded()
            raw_embedding: Any = model.encode(
                _BGE_QUERY_INSTRUCTION + text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            embedding = [float(x) for x in raw_embedding]
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            breaker.record_success()
            health_registry.mark_success("embedding")

            # 写入缓存
            embedding_cache.set(cache_key, embedding)
            return embedding

        except CircuitOpenError:
            logger.error("Embedding 熔断，查询嵌入不可用")
            raise RuntimeError("Embedding 熔断") from None
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("embedding")
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e

    def embed_query_safe(self, text: str) -> list[float] | None:
        """安全版本：失败返回 None 而非抛异常，供降级路径使用"""
        try:
            return self.embed_query(text)
        except Exception as e:
            logger.warning(f"Embedding 安全调用失败: {e}，将走 BM25-only 路径")
            return None


# 全局单例（保持原变量名，调用方 import 无需改动）
vector_embedding_service = LocalEmbeddings()
