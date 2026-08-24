"""混合检索测试：N 路 RRF 已知排名 / 四通道容错 / 降级阶梯 KG_ONLY 档"""

from langchain_core.documents import Document

import app.tools.knowledge_tool as kt
from app.tools.knowledge_tool import (
    _rrf_merge_n,
    retrieve_hybrid,
    retrieve_with_degradation,
)


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"_file_name": f"{text}.md"})


class TestRRFMergeN:
    def test_known_ranking_multi_channel_beats_single(self):
        """两路都命中的文档必须排在单路命中之上（RRF 累加语义）"""
        a = [_doc("共享文档"), _doc("仅A文档")]
        b = [_doc("B路第一名"), _doc("共享文档")]
        merged = _rrf_merge_n([a, b], top_k=3, k=60)

        assert merged[0].page_content == "共享文档"

    def test_three_way_agrees_with_legacy_merge_three(self):
        """N 路泛化与旧三路实现结果一致（回归锚定）"""
        results_a = [(_doc("x1"), 0.9), (_doc("x2"), 0.8)]
        results_b = [(_doc("x2"), 0.7), (_doc("x3"), 0.6)]
        results_c = [(Document(page_content="x3", metadata={}), 5.0), (_doc("x4"), 4.0)]

        legacy = kt._rrf_merge_three(results_a, results_b, results_c, top_k=4)
        generalized = _rrf_merge_n([results_a, results_b, results_c], top_k=4)

        assert [d.page_content for d in legacy] == [d.page_content for d in generalized]

    def test_extra_docs_score_from_rank_zero(self):
        """裸文档通道（如 KG）从 rank0 计分，可与检索路融合去重"""
        lists = [[(_doc("向量命中"), 1.0)]]
        extra = [_doc("图谱子图")]
        merged = _rrf_merge_n(lists, top_k=5, extra_docs=extra)

        assert len(merged) == 2

    def test_dedup_by_first_100_chars(self):
        long_a = "长" * 150 + "尾部A"
        long_b = "长" * 150 + "尾部B"
        merged = _rrf_merge_n([[(_doc(long_a), 1.0)], [(_doc(long_b), 1.0)]], top_k=5)
        assert len(merged) == 1  # 前 100 字符相同视为同一文档

    def test_empty_inputs_yield_empty(self):
        assert _rrf_merge_n([], top_k=3) == []
        assert _rrf_merge_n([[]], top_k=3) == []


class TestRetrieveHybrid:
    async def test_all_channels_fused(self, monkeypatch):
        """四通道各出一条 → 融合后全部出现"""
        monkeypatch.setattr(kt.query_rewriter, "rewrite", _async_return("改写后查询"))
        monkeypatch.setattr(
            kt.health_registry, "is_available", lambda name: True
        )

        class FakeLLM:
            async def ainvoke(self, prompt):
                return type("R", (), {"content": "假设答案"})()

        monkeypatch.setattr(kt.LLMFactory, "create_chat_model", staticmethod(lambda **kw: FakeLLM()))

        class FakeStore:
            def similarity_search_with_score(self, query, k):
                tag = "向量通道" if "改写" in query else "HyDE通道"
                return [(_doc(tag), 1.0)]

        monkeypatch.setattr(kt.vector_store_manager, "get_vector_store", lambda: FakeStore())
        monkeypatch.setattr(
            kt.bm25_retriever, "search",
            lambda q, top_k=5: [(_doc("BM25通道"), 1.0)],
        )
        monkeypatch.setattr(
            kt.graph_retriever, "retrieve", lambda q: [_doc("图谱子图")]
        )
        monkeypatch.setattr(kt.reranker, "rerank", _fake_rerank)

        ctx, docs = await retrieve_hybrid("内存告警", top_k=4)

        contents = {d.page_content for d in docs}
        assert {"向量通道", "HyDE通道", "BM25通道", "图谱子图"} <= contents

    async def test_single_channel_failure_does_not_kill_fusion(self, monkeypatch):
        """BM25 与图通道同时炸 → 其余通道照常返回"""
        monkeypatch.setattr(kt.query_rewriter, "rewrite", _async_return("q"))
        monkeypatch.setattr(kt.health_registry, "is_available", lambda name: False)

        class FakeStore:
            def similarity_search_with_score(self, query, k):
                return [(_doc("幸存向量结果"), 1.0)]

        monkeypatch.setattr(kt.vector_store_manager, "get_vector_store", lambda: FakeStore())

        def _boom(*a, **kw):
            raise RuntimeError("索引损坏")

        monkeypatch.setattr(kt.bm25_retriever, "search", _boom)
        monkeypatch.setattr(kt.graph_retriever, "retrieve", _boom)
        monkeypatch.setattr(kt.reranker, "rerank", _fake_rerank)

        ctx, docs = await retrieve_hybrid("任意查询")
        assert [d.page_content for d in docs] == ["幸存向量结果"]

    async def test_total_failure_degrades_to_plain_retrieval(self, monkeypatch):
        """整体异常 → 降级为普通向量检索（既有兜底契约保持）

        StructuredTool 冻结无法 patch invoke，改为让真实兜底链路自然执行：
        rewrite 抛错触发外层 except；retrieve_knowledge 经被替换的
        vector_store_manager 完成普通检索。
        """
        monkeypatch.setattr(kt.query_rewriter, "rewrite", _async_fail)

        class FakeStore:
            def similarity_search_with_score(self, query, k):
                return [(_doc("兜底文档"), 1.0)]

        monkeypatch.setattr(kt.vector_store_manager, "get_vector_store", lambda: FakeStore())

        ctx, docs = await retrieve_hybrid("任意查询")
        assert "兜底文档" in ctx
        # content_and_artifact 的 invoke 只回传 content 字符串 → 走 (str, []) 分支
        assert docs == []


class TestDegradationLadderKGOnly:
    async def test_kg_only_rung_reached_when_doc_channels_down(self, monkeypatch):
        """Milvus/BM25 全挂但图谱有命中 → 返回 KG_ONLY 档而非静态模板"""
        monkeypatch.setattr(
            kt.health_registry, "is_available", lambda name: False
        )
        # BM25 原始查询也失败
        def _boom(*a, **kw):
            raise RuntimeError("BM25 不可用")

        monkeypatch.setattr(kt.bm25_retriever, "search", _boom)
        kg_doc = Document(
            page_content="[图谱] HighMemoryUsage（内存使用率过高）",
            metadata={"source": "knowledge_graph"},
        )
        monkeypatch.setattr(kt.graph_retriever, "retrieve", lambda q: [kg_doc])

        # 缓存隔离
        monkeypatch.setattr(kt.retrieval_cache, "get", lambda key: None)
        monkeypatch.setattr(kt.retrieval_cache, "set", lambda key, val: None)

        ctx, docs, level = await retrieve_with_degradation("内存使用率过高")

        assert level.value == "kg_only"
        assert "HighMemoryUsage" in ctx

    async def test_full_chain_still_level_zero(self, monkeypatch):
        """健康时 Level 0 走混合四通道"""
        monkeypatch.setattr(kt.health_registry, "is_available", lambda name: True)
        monkeypatch.setattr(kt.retrieval_cache, "get", lambda key: None)
        monkeypatch.setattr(kt.retrieval_cache, "set", lambda key, val: None)

        async def fake_hybrid(query, top_k=None):
            return "混合上下文", [_doc("混合结果")]

        monkeypatch.setattr(kt, "retrieve_hybrid", fake_hybrid)

        ctx, docs, level = await retrieve_with_degradation("查询")

        assert level.value == "none"
        assert docs[0].page_content == "混合结果"


# ──────────────── 辅助 ────────────────


def _async_return(value):
    """monkeypatch 工厂：返回忽略入参、恒返回 value 的异步可调用"""
    async def _call(*_args, **_kwargs):
        return value

    return _call


async def _async_fail(*args, **kwargs):
    raise RuntimeError("改写服务不可用")


async def _fake_rerank(query, documents, top_n):
    return documents[:top_n]


class TestDegradedCacheSemantics:
    """评审修复回归：只有健康档 L0 的结果可入缓存

    降级档是瞬态质量——缓存住 KG_ONLY/BM25 档会在服务恢复后
    继续返回劣化结果长达一个 TTL 窗口。
    """

    def _patch_cache_recording(self, monkeypatch):
        written = []
        monkeypatch.setattr(kt.retrieval_cache, "get", lambda key: None)
        monkeypatch.setattr(
            kt.retrieval_cache, "set", lambda key, val: written.append((key, val))
        )
        return written

    async def test_kg_only_rung_not_cached(self, monkeypatch):
        monkeypatch.setattr(kt.health_registry, "is_available", lambda name: False)

        def _boom(*a, **kw):
            raise RuntimeError("BM25 不可用")

        monkeypatch.setattr(kt.bm25_retriever, "search", _boom)
        kg_doc = Document(page_content="[图谱] HighMemoryUsage", metadata={})
        monkeypatch.setattr(kt.graph_retriever, "retrieve", lambda q: [kg_doc])
        written = self._patch_cache_recording(monkeypatch)

        ctx, docs, level = await retrieve_with_degradation("内存使用率过高")
        assert level.value == "kg_only"
        assert written == []  # 降级档绝不写缓存

    async def test_level_zero_result_is_cached(self, monkeypatch):
        monkeypatch.setattr(kt.health_registry, "is_available", lambda name: True)
        written = self._patch_cache_recording(monkeypatch)

        async def fake_hybrid(query, top_k=None):
            return "混合上下文", [_doc("混合结果")]

        monkeypatch.setattr(kt, "retrieve_hybrid", fake_hybrid)

        ctx, docs, level = await retrieve_with_degradation("查询")
        assert level.value == "none"
        assert len(written) == 1  # 健康档写入缓存，服务恢复前可复用
