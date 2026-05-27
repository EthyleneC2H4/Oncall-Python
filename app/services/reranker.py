"""Rerank 重排服务

使用 DashScope gte-rerank 模型对粗召回结果进行精排：
- 粗召回（向量 + BM25）侧重高召回率，结果可能包含噪声
- Rerank 使用 cross-encoder 对 query-document pair 精确打分
- 精排后 top-k 的精度显著提升
"""

from typing import List

import dashscope
from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.circuit_breaker import get_breaker, BREAKER_RERANK, CircuitOpenError
from app.core.health_registry import health_registry


class Reranker:
    """基于 DashScope gte-rerank 的重排器"""

    def __init__(self, model: str | None = None):
        self.model = model or config.rerank_model

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 3,
    ) -> List[Document]:
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

        breaker = get_breaker(BREAKER_RERANK)
        try:
            breaker.before_call()

            # 提取文档文本
            doc_texts = [doc.page_content for doc in documents]

            # 调用 DashScope Rerank API
            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(documents)),
                api_key=config.dashscope_api_key,
                return_documents=True,
            )

            if response.status_code != 200:
                breaker.record_failure()
                health_registry.mark_failure("dashscope_rerank")
                logger.warning(
                    f"Rerank API 调用失败: {response.code} - {response.message}，"
                    "返回原始排序"
                )
                return documents[:top_n]

            # 解析结果，按 rerank 分数排序
            reranked_docs = []
            for item in response.output.results:
                original_index = item.index
                relevance_score = item.relevance_score
                if original_index < len(documents):
                    doc = documents[original_index]
                    doc.metadata["rerank_score"] = relevance_score
                    reranked_docs.append(doc)

            breaker.record_success()
            health_registry.mark_success("dashscope_rerank")
            logger.info(
                f"Rerank 完成: {len(documents)} 篇 → {len(reranked_docs)} 篇, "
                f"top score={reranked_docs[0].metadata.get('rerank_score', 'N/A') if reranked_docs else 'N/A'}"
            )
            return reranked_docs

        except CircuitOpenError:
            logger.warning("Rerank API 熔断，返回原始排序")
            return documents[:top_n]
        except Exception as e:
            breaker.record_failure()
            health_registry.mark_failure("dashscope_rerank")
            logger.error(f"Rerank 失败: {e}，返回原始排序")
            return documents[:top_n]


# 全局单例
reranker = Reranker()
