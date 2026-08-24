"""知识图谱检索通道 —— 混合检索的第四路（vector / hyde / bm25 之外）

把 KG 从「孤立亮点」接入统一检索面：
    query → 实体抽取（jieba 分词 + 关键词/标签模糊匹配）→ 命中节点
          → 1-hop 子图 → 序列化为 Document（与向量/BM25 结果同构，可进 RRF 融合）

设计约束：
- 纯函数核心（extract_entities / serialize_subgraph）接收显式 nx.DiGraph，
  可用种子图 fixture 独立测试；单例包装只做服务定位
- 图不可达 / 无命中时返回空列表而非抛异常——检索通道失败不拖垮融合
"""

from __future__ import annotations

from typing import Any

import jieba
import networkx as nx
from langchain_core.documents import Document
from loguru import logger

# 静默 jieba 初始化日志（与 bm25_retriever 保持一致）
jieba.setLogLevel(jieba.logging.INFO)

# 单个查询最多命中的种子节点数（防止宽泛查询序列化整个图谱）
_MAX_SEED_NODES = 3


def tokenize_query(query: str) -> list[str]:
    """中文分词 + 去单字噪声（与 BM25 同一套分词口径）"""
    return [t.strip() for t in jieba.lcut(query) if len(t.strip()) > 1]


def extract_entities(graph: nx.DiGraph, query: str) -> list[str]:
    """从查询中抽取能命中图谱节点的实体名（纯函数）

    匹配优先级（先专后泛）：
    1. 节点名 / label 与查询或其 token 的包含关系（大小写不敏感）
    2. 内置运维关键词映射（cpu→HighCPUUsage 等）
    """
    if not query.strip():
        return []

    q_lower = query.lower()
    tokens = [t.lower() for t in tokenize_query(query)]

    matched: list[str] = []
    for node, data in graph.nodes(data=True):
        names = {str(node).lower()}
        label = data.get("label")
        if label:
            names.add(str(label).lower())
        # 节点任一名称被查询原文或任一 token 包含即命中
        if any(name and (name in q_lower or name in tokens) for name in names):
            matched.append(str(node))
        if len(matched) >= _MAX_SEED_NODES:
            break

    if matched:
        return matched

    # 兜底：复用 KG 服务的关键词映射（cpu/内存/慢 等口语词 → 告警节点）
    from app.services.knowledge_graph_service import KnowledgeGraphService

    for keyword, alert in KnowledgeGraphService.ALERT_KEYWORDS.items():
        if keyword in q_lower:
            return [alert]
    return []


def serialize_subgraph(
    graph: nx.DiGraph,
    node: str,
    *,
    max_neighbors: int = 8,
) -> str:
    """把节点的一跳子图序列化为 LLM 友好的文本（纯函数）

    同时收集入边（谁指向我）与出边（我指向谁），按关系类型分组；
    出边邻居附带其直接后继的一层提示（如根因 → 处置动作），
    保证「告警 → 根因 → 处置」的关键推理链在一跳序列化内可见。
    """
    if node not in graph:
        return ""

    data = graph.nodes[node]
    lines = [f"[图谱] {node}（{data.get('label', node)}）"]

    type_label = data.get("type", "")
    if type_label:
        lines[0] += f" 类型:{type_label}"
    for attr in ("level", "threshold", "category", "urgency"):
        if data.get(attr):
            lines[0] += f" {attr}={data[attr]}"

    def _display(name: str) -> str:
        """邻居的展示名：优先中文 label，节点名兜底"""
        label = graph.nodes[name].get("label") if name in graph else None
        return str(label) if label else name

    # 入边：CAUSED_BY x→node 表示 node 的成因
    causes: list[str] = []
    cascades_in: list[str] = []
    for src, _, edge in graph.in_edges(node, data=True):
        relation = edge.get("relation", "")
        reason = edge.get("reason", "")
        suffix = f"（{reason}）" if reason else ""
        shown = _display(src)
        if relation == "CAUSED_BY":
            causes.append(f"{shown}{suffix}")
        elif relation == "MAY_TRIGGER":
            cascades_in.append(f"{shown}{suffix}")
        else:
            causes.append(f"{shown} --{relation}--> {node}")
    if causes:
        lines.append(f"  成因来源: {'; '.join(causes[:max_neighbors])}")
    if cascades_in:
        lines.append(f"  上游触发: {'; '.join(cascades_in[:max_neighbors])}")

    # 出边：按关系分组；根因邻居附带其一层处置动作（关键推理链）
    actions_of_cause: dict[str, list[str]] = {}
    out_relations: dict[str, list[str]] = {}
    for _, dst, edge in graph.out_edges(node, data=True):
        relation = edge.get("relation", "")
        reason = edge.get("reason", "")
        text = f"{_display(dst)}（{reason}）" if reason else _display(dst)
        out_relations.setdefault(relation, []).append(text)
        if relation == "CAUSED_BY":
            resolved = [
                _display(d) for _, d, e in graph.out_edges(dst, data=True)
                if e.get("relation") == "RESOLVED_BY"
            ]
            if resolved:
                actions_of_cause[_display(dst)] = resolved[:4]

    for relation in ("CAUSED_BY", "MAY_TRIGGER", "RESOLVED_BY", "USES"):
        targets = out_relations.get(relation)
        if not targets:
            continue
        zh = {
            "CAUSED_BY": "可能成因",
            "MAY_TRIGGER": "可能级联引发",
            "RESOLVED_BY": "处置动作",
            "USES": "使用工具",
        }.get(relation, relation)
        lines.append(f"  {zh}: {'; '.join(targets[:max_neighbors])}")

    # 根因的处置动作统一附在末尾（避免随关系组重复打印）
    for cause_label, acts in actions_of_cause.items():
        lines.append(f"  {cause_label} 的处置: {'; '.join(acts)}")

    return "\n".join(lines)


class GraphRetriever:
    """KG 检索通道：query → 命中节点子图 → Document 列表"""

    def __init__(self, service: Any = None):
        self._service = service

    @property
    def service(self):
        if self._service is None:
            from app.services.knowledge_graph_service import knowledge_graph_service

            self._service = knowledge_graph_service
        return self._service

    def retrieve(self, query: str) -> list[Document]:
        """检索知识图谱，返回子图文档列表（永不抛异常）

        返回的 Document 与向量/BM25 通道同构（page_content + metadata），
        可直接参与 RRF 融合；metadata 标注 source=knowledge_graph 便于溯源。
        """
        try:
            graph = self.service.graph
            nodes = extract_entities(graph, query)
            if not nodes:
                logger.debug(f"图检索无命中: query='{query[:40]}'")
                return []

            docs: list[Document] = []
            seen_texts: set[str] = set()
            for node in nodes:
                text = serialize_subgraph(graph, node)
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "knowledge_graph",
                            "_file_name": f"知识图谱/{node}",
                            "kg_node": node,
                            "node_type": graph.nodes[node].get("type", ""),
                        },
                    )
                )
            logger.info(f"图检索命中 {len(docs)} 个实体子图: {nodes}")
            return docs
        except Exception as e:  # noqa: BLE001 - 检索通道失败不拖垮混合融合
            logger.warning(f"图检索通道失败（忽略该路）: {e}")
            return []


# 全局单例
graph_retriever = GraphRetriever()
