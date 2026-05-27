"""知识检索工具 - 从向量数据库中检索相关信息

完整 RAG 检索链路：
  Query → Query Rewrite → [Milvus 向量检索 + BM25 关键词检索] → RRF 融合 → Rerank 精排 → 返回

支持 Self-RAG 自我纠正：检索后评估相关性，不相关时过滤。
支持 HyDE（假设文档嵌入）：先生成假设性答案再检索，提升模糊查询召回率。
支持 BM25 稀疏检索：关键词精确匹配，与向量语义检索互补。
支持 Rerank 精排：使用 cross-encoder 对粗召回结果重排，提升 top-k 精度。
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.bm25_retriever import bm25_retriever
from app.services.reranker import reranker
from app.services.query_rewriter import query_rewriter


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

        vector_store = vector_store_manager.get_vector_store()

        # 使用 similarity_search_with_score 获取分数，用于自我评估
        docs_with_scores = vector_store.similarity_search_with_score(
            query, k=config.rag_top_k + 2  # 多检索几条，留出过滤空间
        )

        if not docs_with_scores:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # Self-RAG: 基于分数过滤不相关文档
        relevance_threshold = 800.0  # L2 距离阈值
        filtered_docs = _filter_by_relevance(docs_with_scores, relevance_threshold)

        # 限制最终返回数量
        filtered_docs = filtered_docs[:config.rag_top_k]

        if not filtered_docs:
            logger.warning(f"所有 {len(docs_with_scores)} 条检索结果均不相关 (threshold={relevance_threshold})")
            return "检索到的文档与问题相关性较低，建议换个问法重试。", []

        # 格式化文档为上下文
        context = format_docs(filtered_docs)
        logger.info(f"检索到 {len(filtered_docs)} 个相关文档")
        return context, filtered_docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


async def retrieve_with_hyde(query: str, top_k: int | None = None) -> Tuple[str, List[Document]]:
    """HyDE + Query Rewrite + BM25 + Rerank 全链路增强检索

    完整链路：
    1. Query Rewrite: 改写查询提升检索词质量
    2. HyDE: 生成假设答案做第二路向量检索
    3. BM25: 关键词稀疏检索（第三路）
    4. RRF: 三路结果融合
    5. Rerank: cross-encoder 精排

    Args:
        query: 用户查询
        top_k: 返回的文档数量，默认使用配置

    Returns:
        (格式化上下文, 文档列表)
    """
    from langchain_qwq import ChatQwen

    if top_k is None:
        top_k = config.rag_top_k

    try:
        logger.info(f"增强检索启动: query='{query}'")

        # Step 1: Query Rewrite — 改写查询
        rewritten_query = await query_rewriter.rewrite(query)

        # Step 2: HyDE — 生成假设性答案
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0.3,
        )
        hypothesis = await llm.ainvoke(
            f"请写一段关于以下问题的详细回答（假设你知道答案，直接给出具体内容）：\n{rewritten_query}"
        )
        hypothesis_text = hypothesis.content
        logger.info(f"HyDE 假设答案生成完成，长度: {len(hypothesis_text)}")

        vector_store = vector_store_manager.get_vector_store()

        # Step 3: 三路并行检索
        # 路径A: 用改写后的查询做向量检索
        rewritten_results = vector_store.similarity_search_with_score(
            rewritten_query, k=top_k + 3
        )

        # 路径B: 用假设性答案做向量检索
        hyde_results = vector_store.similarity_search_with_score(
            hypothesis_text, k=top_k + 3
        )

        # 路径C: BM25 关键词检索
        bm25_results = bm25_retriever.search(rewritten_query, top_k=top_k + 3)

        logger.info(
            f"三路检索完成: 向量(rewrite)={len(rewritten_results)}, "
            f"向量(HyDE)={len(hyde_results)}, BM25={len(bm25_results)}"
        )

        # Step 4: RRF 三路融合
        merged_docs = _rrf_merge_three(
            rewritten_results, hyde_results, bm25_results, top_k=top_k + 4
        )

        if not merged_docs:
            return "增强检索未找到相关文档。", []

        # Step 5: Rerank 精排
        reranked_docs = await reranker.rerank(
            query=query,  # 用原始查询做 rerank，保留用户原始意图
            documents=merged_docs,
            top_n=top_k,
        )

        context = format_docs(reranked_docs)
        logger.info(f"增强检索完成: {len(reranked_docs)} 个文档（经 Rerank 精排）")
        return context, reranked_docs

    except Exception as e:
        logger.error(f"增强检索失败: {e}, 降级为普通检索")
        # 降级为普通检索
        result = retrieve_knowledge.invoke({"query": query})
        if isinstance(result, tuple):
            return result
        return str(result), []


async def retrieve_with_rewrite_and_rerank(
    query: str, top_k: int | None = None
) -> Tuple[str, List[Document]]:
    """Query Rewrite + BM25 + Rerank 检索（不含 HyDE，速度更快）

    适用于意图明确的查询（如诊断类），不需要 HyDE 假设生成。
    链路：Query Rewrite → [向量检索 + BM25] → RRF → Rerank

    Args:
        query: 用户查询
        top_k: 返回文档数量

    Returns:
        (格式化上下文, 文档列表)
    """
    if top_k is None:
        top_k = config.rag_top_k

    try:
        logger.info(f"Rewrite+Rerank 检索启动: query='{query}'")

        # Step 1: Query Rewrite
        rewritten_query = await query_rewriter.rewrite(query)

        vector_store = vector_store_manager.get_vector_store()

        # Step 2: 双路检索
        # 路径A: 向量检索（改写后的查询）
        vector_results = vector_store.similarity_search_with_score(
            rewritten_query, k=top_k + 3
        )

        # 路径B: BM25 关键词检索
        bm25_results = bm25_retriever.search(rewritten_query, top_k=top_k + 3)

        logger.info(
            f"双路检索完成: 向量={len(vector_results)}, BM25={len(bm25_results)}"
        )

        # Step 3: RRF 融合
        merged_docs = _rrf_merge(vector_results, bm25_results, top_k=top_k + 4)

        if not merged_docs:
            return "检索未找到相关文档。", []

        # Step 4: Rerank 精排
        reranked_docs = await reranker.rerank(
            query=query,
            documents=merged_docs,
            top_n=top_k,
        )

        context = format_docs(reranked_docs)
        logger.info(f"Rewrite+Rerank 检索完成: {len(reranked_docs)} 个文档")
        return context, reranked_docs

    except Exception as e:
        logger.error(f"Rewrite+Rerank 检索失败: {e}, 降级为普通检索")
        result = retrieve_knowledge.invoke({"query": query})
        if isinstance(result, tuple):
            return result
        return str(result), []


def _filter_by_relevance(
    docs_with_scores: list,
    threshold: float,
) -> List[Document]:
    """Self-RAG 相关性过滤

    Args:
        docs_with_scores: (Document, score) 列表
        threshold: L2 距离阈值

    Returns:
        过滤后的文档列表
    """
    filtered_docs = []
    filtered_out = []

    for doc, score in docs_with_scores:
        if score <= threshold:
            filtered_docs.append(doc)
            logger.debug(f"  保留文档 (score={score:.2f}): {doc.metadata.get('_file_name', '?')}")
        else:
            filtered_out.append((doc, score))
            logger.debug(f"  过滤文档 (score={score:.2f}): {doc.metadata.get('_file_name', '?')}")

    if filtered_out:
        logger.info(
            f"Self-RAG 过滤: {len(docs_with_scores)} 条检索 → "
            f"{len(filtered_docs)} 条保留, {len(filtered_out)} 条过滤"
        )

    return filtered_docs


def _rrf_merge(
    results_a: list,
    results_b: list,
    top_k: int = 3,
    k: int = 60,
) -> List[Document]:
    """Reciprocal Rank Fusion (RRF) 融合两路检索结果

    RRF score = sum(1 / (k + rank_i))

    Args:
        results_a: 第一路检索结果 [(doc, score), ...]
        results_b: 第二路检索结果
        top_k: 返回数量
        k: RRF 参数，默认 60

    Returns:
        融合后的文档列表（去重）
    """
    doc_scores: dict[str, tuple[float, Document]] = {}

    for rank, (doc, _) in enumerate(results_a):
        doc_id = doc.page_content[:100]  # 用前 100 字符作为去重 key
        rrf_score = 1.0 / (k + rank)
        if doc_id in doc_scores:
            doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, doc)
        else:
            doc_scores[doc_id] = (rrf_score, doc)

    for rank, (doc, _) in enumerate(results_b):
        doc_id = doc.page_content[:100]
        rrf_score = 1.0 / (k + rank)
        if doc_id in doc_scores:
            doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, doc)
        else:
            doc_scores[doc_id] = (rrf_score, doc)

    # 按 RRF score 排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in sorted_docs[:top_k]]


def _rrf_merge_three(
    results_a: list,
    results_b: list,
    results_c: list,
    top_k: int = 5,
    k: int = 60,
) -> List[Document]:
    """RRF 融合三路检索结果

    Args:
        results_a: 第一路（向量检索-改写查询）
        results_b: 第二路（向量检索-HyDE）
        results_c: 第三路（BM25）
        top_k: 返回数量
        k: RRF 参数

    Returns:
        融合后的文档列表
    """
    doc_scores: dict[str, tuple[float, Document]] = {}

    for results in [results_a, results_b, results_c]:
        for rank, item in enumerate(results):
            # 兼容 (doc, score) 元组格式
            doc = item[0] if isinstance(item, tuple) else item
            doc_id = doc.page_content[:100]
            rrf_score = 1.0 / (k + rank)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, doc)
            else:
                doc_scores[doc_id] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in sorted_docs[:top_k]]


def format_docs(docs: List[Document]) -> str:
    """格式化文档列表为上下文文本"""
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"

        # 如果有 rerank 分数，展示
        rerank_score = metadata.get("rerank_score")
        if rerank_score is not None:
            formatted += f"\n相关度: {rerank_score:.4f}"

        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
