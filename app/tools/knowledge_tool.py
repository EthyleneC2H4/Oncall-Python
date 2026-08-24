"""知识检索工具 - 从向量数据库中检索相关信息

完整 RAG 检索链路：
  Query → Query Rewrite → [Milvus 向量检索 + BM25 关键词检索] → RRF 融合 → Rerank 精排 → 返回

支持 Self-RAG 自我纠正：检索后评估相关性，不相关时过滤。
支持 HyDE（假设文档嵌入）：先生成假设性答案再检索，提升模糊查询召回率。
支持 BM25 稀疏检索：关键词精确匹配，与向量语义检索互补。
支持 Rerank 精排：使用 cross-encoder 对粗召回结果重排，提升 top-k 精度。
支持混合四通道检索（P4）：[向量, HyDE, BM25, 知识图谱] → N 路 RRF 融合。
"""


from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.core.cache import make_cache_key, retrieval_cache
from app.core.degradation import DegradationLevel
from app.core.health_registry import health_registry
from app.core.llm_factory import LLMFactory
from app.services.bm25_retriever import bm25_retriever
from app.services.graph_retriever import graph_retriever
from app.services.query_rewriter import query_rewriter
from app.services.reranker import reranker
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> tuple[str, list[Document]]:
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
        filtered_docs = filtered_docs[: config.rag_top_k]

        if not filtered_docs:
            logger.warning(
                f"所有 {len(docs_with_scores)} 条检索结果均不相关 (threshold={relevance_threshold})"
            )
            return "检索到的文档与问题相关性较低，建议换个问法重试。", []

        # 格式化文档为上下文
        context = format_docs(filtered_docs)
        logger.info(f"检索到 {len(filtered_docs)} 个相关文档")
        return context, filtered_docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


async def retrieve_with_hyde(query: str, top_k: int | None = None) -> tuple[str, list[Document]]:
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

    if top_k is None:
        top_k = config.rag_top_k

    try:
        logger.info(f"增强检索启动: query='{query}'")

        # Step 1: Query Rewrite — 改写查询
        rewritten_query = await query_rewriter.rewrite(query)

        # Step 2: HyDE — 生成假设性答案
        llm = LLMFactory.create_chat_model(
            streaming=False,
            model=config.rag_model,
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
            str(rewritten_query), k=top_k + 3
        )

        # 路径B: 用假设性答案做向量检索
        hyde_results = vector_store.similarity_search_with_score(
            str(hypothesis_text), k=top_k + 3
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
) -> tuple[str, list[Document]]:
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
        vector_results = vector_store.similarity_search_with_score(rewritten_query, k=top_k + 3)

        # 路径B: BM25 关键词检索
        bm25_results = bm25_retriever.search(rewritten_query, top_k=top_k + 3)

        logger.info(f"双路检索完成: 向量={len(vector_results)}, BM25={len(bm25_results)}")

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


async def retrieve_hybrid(query: str, top_k: int | None = None) -> tuple[str, list[Document]]:
    """四通道混合检索（P4）：[向量, HyDE, BM25, 知识图谱] → N 路 RRF → Rerank

    在 rewrite+rerank 链路之上新增两路：
    - HyDE 假设答案向量检索（LLM 可用时）
    - KG 一跳子图检索（graph_retriever；结构化因果知识，与文档语义互补）

    单通道失败只记警告并丢弃该路，其余通道照常融合——
    检索面整体可用性高于任何单路的最优性。

    Returns:
        (格式化上下文, 文档列表)
    """
    if top_k is None:
        top_k = config.rag_top_k

    try:
        logger.info(f"混合四通道检索启动: query='{query}'")

        rewritten_query = await query_rewriter.rewrite(query)
        vector_store = vector_store_manager.get_vector_store()

        channel_lists: list[list] = []

        # 通道 1：改写查询向量检索
        try:
            vector_results = vector_store.similarity_search_with_score(
                str(rewritten_query), k=top_k + 3
            )
            channel_lists.append(vector_results)
        except Exception as e:  # noqa: BLE001 - 单通道失败不拖垮融合
            logger.warning(f"混合检索·向量通道失败（跳过）: {e}")

        # 通道 2：HyDE 假设答案向量检索（需 LLM）
        if health_registry.is_available("llm"):
            try:
                llm = LLMFactory.create_chat_model(
                    streaming=False, model=config.rag_model, temperature=0.3
                )
                hypothesis = await llm.ainvoke(
                    "请写一段关于以下问题的详细回答"
                    f"（假设你知道答案，直接给出具体内容）：\n{rewritten_query}"
                )
                hyde_results = vector_store.similarity_search_with_score(
                    str(hypothesis.content), k=top_k + 3
                )
                channel_lists.append(hyde_results)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"混合检索·HyDE 通道失败（跳过）: {e}")
        else:
            logger.info("LLM 不可用，HyDE 通道跳过")

        # 通道 3：BM25 关键词检索
        try:
            bm25_results = bm25_retriever.search(str(rewritten_query), top_k=top_k + 3)
            channel_lists.append(bm25_results)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"混合检索·BM25 通道失败（跳过）: {e}")

        # 通道 4：知识图谱一跳子图
        try:
            kg_docs = graph_retriever.retrieve(query)
        except Exception as e:  # noqa: BLE001 - 单通道失败不拖垮融合
            logger.warning(f"混合检索·图谱通道失败（跳过）: {e}")
            kg_docs = []

        merged_docs = _rrf_merge_n(channel_lists, extra_docs=kg_docs, top_k=top_k + 4)
        if not merged_docs:
            return "混合检索未找到相关文档。", []

        reranked_docs = await reranker.rerank(
            query=query, documents=merged_docs, top_n=top_k
        )
        context = format_docs(reranked_docs)
        logger.info(f"混合检索完成: 融合 {len(merged_docs)} → 精排 {len(reranked_docs)}")
        return context, reranked_docs

    except Exception as e:
        logger.error(f"混合检索失败: {e}, 降级为普通检索")
        result = retrieve_knowledge.invoke({"query": query})
        if isinstance(result, tuple):
            return result
        return str(result), []


def _filter_by_relevance(
    docs_with_scores: list,
    threshold: float,
) -> list[Document]:
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
) -> list[Document]:
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


def _rrf_merge_n(
    result_lists: list[list],
    top_k: int = 5,
    k: int = 60,
    extra_docs: list[Document] | None = None,
) -> list[Document]:
    """RRF 泛化：融合任意 N 路检索结果（P4）

    RRF score = Σ 1 / (k + rank_i)，文档按前 100 字符去重；
    extra_docs 作为无分数的第 N+1 路（如 KG 子图文档）从 rank 0 计分。

    Args:
        result_lists: 每路为 [(Document, score), ...] 或 [Document, ...]
        top_k: 返回数量
        k: RRF 平滑参数
        extra_docs: 追加通道的裸文档列表

    Returns:
        融合去重后的文档列表
    """
    doc_scores: dict[str, tuple[float, Document]] = {}

    def _add(doc: Document, rank: int) -> None:
        doc_id = doc.page_content[:100]
        rrf_score = 1.0 / (k + rank)
        if doc_id in doc_scores:
            doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, doc)
        else:
            doc_scores[doc_id] = (rrf_score, doc)

    for results in result_lists:
        for rank, item in enumerate(results):
            doc = item[0] if isinstance(item, tuple) else item
            _add(doc, rank)

    for rank, doc in enumerate(extra_docs or []):
        _add(doc, rank)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:top_k]]


def _rrf_merge_three(
    results_a: list,
    results_b: list,
    results_c: list,
    top_k: int = 5,
    k: int = 60,
) -> list[Document]:
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


async def retrieve_with_degradation(
    query: str,
    top_k: int | None = None,
) -> tuple[str, list[Document], DegradationLevel]:
    """按降级层级检索，逐级降级直到获得结果

    Level 0: 混合四通道 (向量+HyDE+BM25+KG+RRF+Rerank)
    Level 1: 无 Rerank (向量+BM25+RRF)
    Level 2: BM25-only (带改写)
    Level 3: BM25-only (原始查询)
    Level 4: KG-only (知识图谱子图，文档检索全不可用时)
    Level 5: 缓存
    Level 6: 静态兜底

    Returns:
        (格式化上下文, 原始文档列表, 降级等级)
    """
    if top_k is None:
        top_k = config.rag_top_k

    # 检查缓存（只有健康档 L0 的结果才写入——降级档是瞬态质量，
    # 缓存会把 KG_ONLY/BM25 劣化档在服务恢复后仍钉住一个 TTL 窗口）
    cache_key = make_cache_key("retrieval", query, str(top_k))
    cached = retrieval_cache.get(cache_key)
    if cached is not None:
        ctx, docs = cached
        return ctx, docs, DegradationLevel.CACHED

    milvus_ok = health_registry.is_available("milvus") and health_registry.is_available("embedding")
    rerank_ok = health_registry.is_available("rerank")
    llm_ok = health_registry.is_available("llm")

    # Level 0: 混合四通道完整链路
    if milvus_ok and rerank_ok:
        try:
            if llm_ok:
                ctx, docs = await retrieve_hybrid(query, top_k)
            else:
                ctx, docs = retrieve_knowledge.invoke({"query": query})
                if isinstance(ctx, str) and not docs:
                    raise ValueError("empty")
                docs = docs if isinstance(docs, list) else []
            if docs:
                retrieval_cache.set(cache_key, (ctx, docs))
                return ctx, docs, DegradationLevel.NONE
        except Exception as e:
            logger.warning(f"完整链路检索失败: {e}")

    # Level 1: 无 Rerank
    if milvus_ok:
        try:
            vector_store = vector_store_manager.get_vector_store()
            rewritten = query
            if llm_ok:
                try:
                    rewritten = await query_rewriter.rewrite(query)
                except Exception:
                    pass
            vector_results = vector_store.similarity_search_with_score(rewritten, k=top_k + 3)
            bm25_results = bm25_retriever.search(rewritten, top_k=top_k + 3)
            merged = _rrf_merge(vector_results, bm25_results, top_k=top_k)
            if merged:
                ctx = format_docs(merged)
                return ctx, merged, DegradationLevel.NO_RERANK
        except Exception as e:
            logger.warning(f"无 Rerank 检索失败: {e}")

    # Level 2: BM25-only (带改写)
    if llm_ok:
        try:
            rewritten = await query_rewriter.rewrite(query)
            bm25_results = bm25_retriever.search(rewritten, top_k=top_k)
            docs = [doc for doc, _ in bm25_results]
            if docs:
                ctx = format_docs(docs)
                return ctx, docs, DegradationLevel.BM25_ONLY
        except Exception as e:
            logger.warning(f"BM25+改写检索失败: {e}")

    # Level 3: BM25-only (原始查询)
    try:
        bm25_results = bm25_retriever.search(query, top_k=top_k)
        docs = [doc for doc, _ in bm25_results]
        if docs:
            ctx = format_docs(docs)
            return ctx, docs, DegradationLevel.BM25_RAW
    except Exception as e:
        logger.warning(f"BM25 原始检索失败: {e}")

    # Level 4: KG-only —— 文档检索全不可用，但图谱子图仍可提供结构化因果知识
    try:
        kg_docs = graph_retriever.retrieve(query)
        if kg_docs:
            ctx = format_docs(kg_docs)
            return ctx, kg_docs, DegradationLevel.KG_ONLY
    except Exception as e:  # noqa: BLE001 - 图检索也失败则继续走静态兜底
        logger.warning(f"KG-only 检索失败: {e}")

    # Level 5: 静态兜底
    logger.warning("所有检索路径均失败，返回静态兜底")
    return "没有找到相关信息，请稍后重试。", [], DegradationLevel.TEMPLATE


def format_docs(docs: list[Document]) -> str:
    """格式化文档列表为上下文文本

    Parent-Child 策略：
    - 如果 metadata 中有 parent_content，使用 parent 内容（更完整的上下文）
    - 按 parent_id 去重，同一 parent 只输出一次
    """
    formatted_parts: list[str] = []
    seen_parents: set[str] = set()

    for doc in docs:
        metadata = doc.metadata
        parent_id = metadata.get("parent_id", "")

        # 按 parent_id 去重：同一个 parent 章节只输出一次
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)

        source = metadata.get("_file_name", "未知来源")

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        idx = len(formatted_parts) + 1
        formatted = f"【参考资料 {idx}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"

        # chunk 类型标注
        chunk_type = metadata.get("chunk_type", "")
        if chunk_type and chunk_type != "general":
            type_label = {
                "procedure": "排查步骤",
                "root_cause": "原因分析",
                "action": "处置方案",
                "alert_info": "告警信息",
                "verification": "验证步骤",
            }.get(chunk_type, chunk_type)
            formatted += f"\n类型: {type_label}"

        # 如果有 rerank 分数，展示
        rerank_score = metadata.get("rerank_score")
        if rerank_score is not None:
            formatted += f"\n相关度: {rerank_score:.4f}"

        # Parent-Child: 优先使用 parent_content（完整章节上下文）
        content = metadata.get("parent_content", "") or doc.page_content
        formatted += f"\n内容:\n{content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
