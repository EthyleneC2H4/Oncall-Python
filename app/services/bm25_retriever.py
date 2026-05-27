"""BM25 关键词检索服务

基于 BM25Okapi 实现稀疏关键词检索，与 Milvus 向量稠密检索互补：
- 向量检索擅长语义相似度，但可能漏掉精确关键词匹配
- BM25 擅长精确关键词匹配（告警名称、错误码、服务名等）
- 两者融合（RRF）可显著提升召回率

使用 jieba 对中文文本进行分词。

降级增强：
- 支持磁盘持久化索引（pickle），Milvus 不可用时从磁盘恢复
- 可独立运行，不依赖 Milvus 连接
"""

import os
import pickle
import threading
from typing import List, Tuple

import jieba
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.milvus_client import milvus_manager

# 静默 jieba 初始化日志
jieba.setLogLevel(jieba.logging.INFO)

# 持久化路径
_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_INDEX_PATH = os.path.join(_INDEX_DIR, "bm25_index.pkl")


class BM25Retriever:
    """BM25 稀疏关键词检索器"""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._lock = threading.Lock()
        self._initialized = False

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 + 去停用词"""
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if len(t.strip()) > 1]

    def _load_documents_from_milvus(self) -> List[Document]:
        """从 Milvus 加载所有文档文本"""
        try:
            collection = milvus_manager.get_collection()
            collection.load()

            results = collection.query(
                expr="id != ''",
                output_fields=["content", "metadata"],
                limit=10000,
            )

            documents = []
            for item in results:
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata if isinstance(metadata, dict) else {},
                    ))

            logger.info(f"BM25: 从 Milvus 加载 {len(documents)} 篇文档")
            return documents

        except Exception as e:
            logger.error(f"BM25: 从 Milvus 加载文档失败: {e}")
            return []

    def build_index(self, documents: List[Document] | None = None):
        """构建 BM25 索引

        Args:
            documents: 文档列表。如果为 None，自动从 Milvus 加载
        """
        with self._lock:
            if documents is None:
                documents = self._load_documents_from_milvus()

            if not documents:
                # Milvus 不可用时尝试从磁盘恢复
                if self._load_from_disk():
                    return
                logger.warning("BM25: 没有文档可供索引")
                return

            self._documents = documents
            self._tokenized_corpus = [
                self._tokenize(doc.page_content) for doc in documents
            ]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self._initialized = True
            logger.info(f"BM25: 索引构建完成，共 {len(documents)} 篇文档")

            # 持久化到磁盘
            self._save_to_disk()

    def _save_to_disk(self) -> None:
        """将索引持久化到磁盘"""
        try:
            os.makedirs(_INDEX_DIR, exist_ok=True)
            data = {
                "documents": self._documents,
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(_INDEX_PATH, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"BM25: 索引已持久化到 {_INDEX_PATH}")
        except Exception as e:
            logger.warning(f"BM25: 索引持久化失败: {e}")

    def _load_from_disk(self) -> bool:
        """从磁盘恢复索引"""
        if not os.path.exists(_INDEX_PATH):
            return False
        try:
            with open(_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._tokenized_corpus = data["tokenized_corpus"]
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
                self._initialized = True
                logger.info(f"BM25: 从磁盘恢复索引，共 {len(self._documents)} 篇文档")
                return True
        except Exception as e:
            logger.warning(f"BM25: 从磁盘恢复索引失败: {e}")
        return False

    def _ensure_index(self):
        """确保索引已构建（懒初始化）"""
        if not self._initialized:
            self.build_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            [(Document, score), ...] 按分数降序排列
        """
        self._ensure_index()

        if not self._bm25 or not self._documents:
            logger.warning("BM25: 索引为空，无法检索")
            return []

        try:
            tokenized_query = self._tokenize(query)
            if not tokenized_query:
                return []

            scores = self._bm25.get_scores(tokenized_query)

            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            results = []
            for idx, score in scored_indices:
                if score > 0:
                    results.append((self._documents[idx], float(score)))

            logger.info(f"BM25 检索: query='{query[:50]}', 命中 {len(results)} 篇")
            return results

        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

    def invalidate(self):
        """使索引失效，下次检索时重建（文档更新后调用）"""
        with self._lock:
            self._initialized = False
            self._bm25 = None
            self._documents = []
            self._tokenized_corpus = []
            logger.info("BM25: 索引已失效，将在下次检索时重建")


# 全局单例
bm25_retriever = BM25Retriever()
