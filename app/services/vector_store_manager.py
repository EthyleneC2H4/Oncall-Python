"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.bm25_retriever import bm25_retriever
from app.services.vector_embedding_service import vector_embedding_service

# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器（惰性：首次使用时才连接 Milvus）"""
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            # 必须在 PyMilvus / langchain_milvus 访问 Collection 之前建立连接，
            # 否则会出现 ConnectionNotExistException: should create connection first.
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector",  # 向量存储到 vector 字段
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid

            start_time = time.time()

            # 为每个文档生成唯一 id（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            result_ids = self.get_vector_store().add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed / len(documents):.2f}秒/个"
            )

            # 文档变更后使 BM25 索引失效，下次检索时自动重建
            bm25_retriever.invalidate()

            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例（惰性初始化：首次调用才连接 Milvus）

        惰性化的原因：模块导入期不触发网络连接，应用可在 Milvus
        不可用时以降级模式启动（见 main.py lifespan），测试也可离线导入。

        Returns:
            Milvus: VectorStore 实例

        Raises:
            RuntimeError: Milvus 连接失败时抛出
        """
        if self.vector_store is None:
            self._initialize_vector_store()
        assert self.vector_store is not None
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            docs = self.get_vector_store().similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
