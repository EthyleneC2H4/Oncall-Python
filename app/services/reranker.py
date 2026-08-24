"""Rerank 重排服务 - 本地 cross-encoder 模型

使用本地 BGE reranker（FlagEmbedding，默认 BAAI/bge-reranker-base）对粗召回结果精排：
- 粗召回（向量 + BM25）侧重高召回率，结果可能包含噪声
- Rerank 使用 cross-encoder 对 query-document pair 精确打分
- 精排后 top-k 的精度显著提升

模型懒加载；可通过 config.rerank_enabled=False 关闭（直接返回原始排序）。
"""

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.circuit_breaker import BREAKER_RERANK, CircuitOpenError, get_breaker
from app.core.health_registry import health_registry


class Reranker:
    """基于本地 FlagEmbedding cross-encoder 的重排器"""

    def __init__(self, model: str | None = None):
        self.model = model or config.rerank_model
        self._model = None  # 懒加载

    def _ensure_loaded(self):
        """懒加载底层重排模型"""
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info(f"加载本地重排模型 {self.model}...")
            self._model = FlagReranker(self.model, use_fp16=False)
            logger.info("本地重排模型加载完成")
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 3,
    ) -> list[Document]:
        """对文档列表进行重排

        Args:
            query: 用户查询
            documents: 候选文档列表
            top_n: 返回的文档数量

        Returns:
            重排后的文档列表（按相关性降序）
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return documents

        if not config.rerank_enabled:
            logger.debug("Rerank 已通过配置关闭，返回原始排序")
            return documents[:top_n]

        breaker = get_breaker(BREAKER_RERANK)
        try:
            breaker.before_call()

            doc_texts = [doc.page_content for doc in documents]
            model = self._ensure_loaded()
            scores = model.compute_score(
                [[query, text] for text in doc_texts],
                normalize=True,
            )
            # 单文档时 compute_score 返回标量
            if not isinstance(scores, list):
                scores = [scores]

            ranked = sorted(
                zip(scores, range(len(documents)), strict=False),
                key=lambda pair: pair[0],
                reverse=True,
            )

            reranked_docs = []
            for score, original_index in ranked[: min(top_n, len(documents))]:
                doc = documents[original_index]
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)

            breaker.record_success()
            health_registry.mark_success("rerank")
            logger.info(
                f"Rerank 完成: {len(documents)} 篇 → {len(reranked_docs)} 篇, "
                f"top score={reranked_docs[0].metadata.get('rerank_score', 'N/A') if reranked_docs else 'N/A'}"
            )
            return reranked_docs

        except CircuitOpenError:
            logger.warning("Rerank 熔断，返回原始排序")
            return documents[:top_n]
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("rerank")
            logger.error(f"Rerank 失败: {e}，返回原始排序")
            return documents[:top_n]


# 全局单例
reranker = Reranker()
