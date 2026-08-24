"""图检索通道测试：实体抽取 / 子图序列化 / 通道容错"""

import networkx as nx
import pytest

from app.services.graph_retriever import (
    GraphRetriever,
    extract_entities,
    serialize_subgraph,
    tokenize_query,
)


@pytest.fixture
def seed_graph():
    """最小种子图：告警 → 根因 → 处置 + 级联边"""
    g = nx.DiGraph()
    g.add_node("HighCPUUsage", type="alert", label="CPU使用率过高", level="critical")
    g.add_node("HighMemoryUsage", type="alert", label="内存使用率过高", level="critical")
    g.add_node("MemoryLeak", type="root_cause", label="内存泄漏", category="code")
    g.add_node("RestartInstance", type="action", label="重启实例", urgency="immediate")
    g.add_node("DumpMemory", type="action", label="导出内存快照分析", urgency="short_term")
    g.add_node("SlowResponse", type="alert", label="响应变慢", level="warning")

    g.add_edge("HighMemoryUsage", "MemoryLeak", relation="CAUSED_BY")
    g.add_edge("MemoryLeak", "RestartInstance", relation="RESOLVED_BY")
    g.add_edge("MemoryLeak", "DumpMemory", relation="RESOLVED_BY")
    g.add_edge("HighMemoryUsage", "SlowResponse",
               relation="MAY_TRIGGER", reason="内存压力导致处理变慢")
    return g


class TestExtractEntities:
    def test_label_substring_match(self, seed_graph):
        nodes = extract_entities(seed_graph, "内存使用率过高怎么处理")
        assert "HighMemoryUsage" in nodes

    def test_token_match(self, seed_graph):
        """分词后的 token 命中节点 label"""
        nodes = extract_entities(seed_graph, "服务出现内存泄漏")
        assert "MemoryLeak" in nodes or "HighMemoryUsage" in nodes

    def test_keyword_fallback(self, seed_graph):
        """口语词经 KG 服务关键词映射兜底（cpu→HighCPUUsage）"""
        # 种子图没有 cpu 关键字节点名，走 ALERT_KEYWORDS 兜底
        nodes = extract_entities(seed_graph, "cpu 很高")
        assert nodes == ["HighCPUUsage"]

    def test_no_match_returns_empty(self, seed_graph):
        assert extract_entities(seed_graph, "今天天气不错") == []

    def test_blank_query_returns_empty(self, seed_graph):
        assert extract_entities(seed_graph, "  ") == []

    def test_max_seed_nodes_capped(self, seed_graph):
        """命中数有上限，防止宽泛查询序列化整个图谱

        注意命中方向是「节点名 ⊆ 查询」：查询必须包含多个节点名
        才会真正触达 _MAX_SEED_NODES 截断。
        """
        g = nx.DiGraph()
        for i in range(10):
            g.add_node(f"Alert{i}", type="alert", label=f"告警{i}")
        # 查询显式包含 6 个节点 label → 无截断时应命中 6 个
        query = "告警0 告警1 告警2 告警3 告警4 告警5"
        nodes = extract_entities(g, query)
        assert len(nodes) == 3  # 恰好截断到上限，而非宽松的 <=


class TestSerializeSubgraph:
    def test_full_chain_visible_in_one_hop(self, seed_graph):
        """告警节点一跳序列化必须能看到 成因→根因 及其处置动作（邻居展示中文 label）"""
        text = serialize_subgraph(seed_graph, "HighMemoryUsage")
        assert "CPU使用率过高" not in text  # 只序列化命中的节点
        assert "内存使用率过高" in text
        assert "内存泄漏" in text and "MemoryLeak" not in text  # 邻居用 label 展示
        # RESOLVED_BY 的二层提示（根因的处置），且只附一次（不随关系组重复）
        assert "导出内存快照分析" in text
        assert text.count("的处置") == 1
        # 级联边带原因
        assert "内存压力导致处理变慢" in text

    def test_root_cause_node_shows_actions(self, seed_graph):
        text = serialize_subgraph(seed_graph, "MemoryLeak")
        assert "处置动作" in text
        assert "重启实例" in text
        assert "成因来源: 内存使用率过高" in text  # 入边邻居同样 label 展示

    def test_unknown_node_returns_empty(self, seed_graph):
        assert serialize_subgraph(seed_graph, "NoSuchNode") == ""

    def test_neighbor_cap(self, seed_graph):
        """邻居数量截断：max_neighbors 控制输出规模"""
        g = nx.DiGraph()
        g.add_node("Hub", type="root_cause", label="枢纽")
        for i in range(15):
            g.add_node(f"A{i}", type="action", label=f"处置{i}")
            g.add_edge("Hub", f"A{i}", relation="RESOLVED_BY")
        text = serialize_subgraph(g, "Hub", max_neighbors=5)
        shown = [i for i in range(15) if f"处置{i}" in text]
        assert len(shown) == 5  # 恰好保留前 5 个邻居


class TestGraphRetriever:
    def test_retrieve_returns_documents(self, seed_graph):
        class FakeService:
            graph = seed_graph

        retriever = GraphRetriever(service=FakeService())
        docs = retriever.retrieve("内存使用率过高")

        assert len(docs) >= 1
        (doc,) = docs
        assert doc.metadata["source"] == "knowledge_graph"
        assert doc.metadata["kg_node"] == "HighMemoryUsage"
        assert "内存泄漏" in doc.page_content

    def test_retrieve_empty_on_no_match(self, seed_graph):
        class FakeService:
            graph = seed_graph

        assert GraphRetriever(service=FakeService()).retrieve("天气") == []

    def test_retrieve_never_raises_on_broken_service(self):
        """服务抛异常 → 返回空列表而非炸掉融合链路"""
        class BrokenService:
            @property
            def graph(self):
                raise RuntimeError("图谱加载失败")

        assert GraphRetriever(service=BrokenService()).retrieve("cpu 高") == []

    def test_duplicate_serialization_deduped(self, seed_graph):
        """同一节点的重复命中只产出一个 Document"""
        class FakeService:
            graph = seed_graph

        retriever = GraphRetriever(service=FakeService())
        docs = retriever.retrieve("内存使用率过高 内存泄漏")
        texts = [d.page_content for d in docs]
        assert len(texts) == len(set(texts))


def test_tokenize_query_drops_single_chars():
    tokens = tokenize_query("CPU 使用率过高了")
    assert all(len(t) > 1 for t in tokens)
