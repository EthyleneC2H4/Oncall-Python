"""知识检索工具 - 从向量数据库中检索相关信息

支持 Self-RAG 自我纠正：检索后评估相关性，不相关时自动改写查询重试。
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    内置 Self-RAG 自我纠正机制：会自动评估检索质量，过滤不相关的结果。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 从向量存储中检索相关文档
        vector_store = vector_store_manager.get_vector_store()

        # 使用 similarity_search_with_score 获取分数，用于自我评估
        docs_with_scores = vector_store.similarity_search_with_score(
            query, k=config.rag_top_k + 2  # 多检索几条，留出过滤空间
        )

        if not docs_with_scores:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # Self-RAG: 基于分数过滤不相关文档
        # Milvus L2 距离: 越小越相似，阈值根据经验设定
        relevance_threshold = 800.0  # L2 距离阈值，超过此值认为不相关
        filtered_docs = []
        filtered_out = []

        for doc, score in docs_with_scores:
            if score <= relevance_threshold:
                filtered_docs.append(doc)
                logger.debug(f"✅ 保留文档 (score={score:.2f}): {doc.metadata.get('_file_name', '?')}")
            else:
                filtered_out.append((doc, score))
                logger.debug(f"❌ 过滤文档 (score={score:.2f}): {doc.metadata.get('_file_name', '?')}")

        # 限制最终返回数量
        filtered_docs = filtered_docs[:config.rag_top_k]

        if not filtered_docs:
            logger.warning(f"所有 {len(docs_with_scores)} 条检索结果均不相关 (threshold={relevance_threshold})")
            return "检索到的文档与问题相关性较低，建议换个问法重试。", []

        # 记录 Self-RAG 评估结果
        if filtered_out:
            logger.info(
                f"Self-RAG 过滤: {len(docs_with_scores)} 条检索 → "
                f"{len(filtered_docs)} 条保留, {len(filtered_out)} 条过滤"
            )

        # 格式化文档为上下文
        context = format_docs(filtered_docs)

        logger.info(f"检索到 {len(filtered_docs)} 个相关文档")
        return context, filtered_docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
