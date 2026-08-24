"""检索结果融合服务（P5 自 knowledge_tool.py 机械迁出）

多路检索的 RRF（Reciprocal Rank Fusion）融合纯函数：
只做「排名 → 融合得分 → 去重排序」，不发起任何 IO，
便于离线单测与复用（knowledge_tool 经垫片别名继续使用旧名）。

去重 key 约定：文档 page_content 前 100 字符——同一文档被
多个通道召回时应视为同一候选。
"""

from langchain_core.documents import Document


def rrf_merge(
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


def rrf_merge_n(
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


def rrf_merge_three(
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
