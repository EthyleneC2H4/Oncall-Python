"""知识图谱接口 - 提供告警关联分析、级联预测和图谱更新的 REST API"""

from typing import Optional

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from app.services.knowledge_graph_service import knowledge_graph_service
from app.services.kg_extractor import kg_extractor

router = APIRouter()


@router.get("/kg/stats")
async def get_kg_stats():
    """获取知识图谱统计信息"""
    stats = knowledge_graph_service.get_graph_stats()
    return {"code": 200, "data": stats}


@router.get("/kg/analyze/{alert_keyword}")
async def analyze_alert(alert_keyword: str):
    """分析指定告警的根因、处置建议和级联风险

    Args:
        alert_keyword: 告警关键词（支持中英文，如 CPU、内存、disk）
    """
    logger.info(f"KG API: 分析告警 '{alert_keyword}'")
    analysis = knowledge_graph_service.get_alert_analysis(alert_keyword)
    return {"code": 200, "data": analysis}


@router.get("/kg/cascade/{alert_keyword}")
async def predict_cascade(alert_keyword: str):
    """预测告警级联链路

    Args:
        alert_keyword: 告警关键词
    """
    logger.info(f"KG API: 级联预测 '{alert_keyword}'")
    prediction = knowledge_graph_service.get_cascade_prediction(alert_keyword)
    return {"code": 200, "data": {"prediction": prediction}}


class ExtractRequest(BaseModel):
    """文档三元组抽取请求"""
    text: str = Field(description="文档文本内容")


class IncidentLearnRequest(BaseModel):
    """故障事件学习请求"""
    incident_id: str = Field(default="manual", description="事件ID")
    alert_type: str = Field(description="告警类型")
    root_cause: str = Field(default="", description="根因")
    resolution: str = Field(default="", description="解决方案")
    cascade_alerts: list[str] = Field(default_factory=list, description="级联告警列表")


@router.post("/kg/extract")
async def extract_and_update(request: ExtractRequest):
    """从文档中自动抽取实体关系并更新知识图谱

    使用 LLM 从输入文本中提取运维领域的三元组，
    然后增量更新到知识图谱中。

    Args:
        request: 包含文档文本的请求
    """
    logger.info(f"KG API: 从文档抽取三元组，文本长度: {len(request.text)}")
    triples = await kg_extractor.extract_from_text(request.text)

    if not triples:
        return {"code": 200, "data": {"triples_extracted": 0, "message": "未抽取到有效三元组"}}

    # 更新图谱
    stats = knowledge_graph_service.update_from_triples(triples)

    return {
        "code": 200,
        "data": {
            "triples_extracted": len(triples),
            "triples": [t.model_dump() for t in triples],
            **stats,
        }
    }


@router.post("/kg/learn-incident")
async def learn_from_incident(request: IncidentLearnRequest):
    """从已解决的故障事件中学习，增量更新知识图谱

    Args:
        request: 故障事件记录
    """
    logger.info(f"KG API: 从事件学习 {request.incident_id}")
    stats = knowledge_graph_service.update_from_incident(request.model_dump())
    return {"code": 200, "data": stats}


@router.get("/kg/graph")
async def get_full_graph():
    """获取完整的图谱数据（用于前端可视化）

    返回节点和边的列表，适配 vis-network / D3.js 等前端图可视化库。
    """
    import networkx as nx

    graph = knowledge_graph_service.graph

    nodes = []
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        # 不同类型节点的颜色映射
        color_map = {
            "alert": "#e74c3c",
            "root_cause": "#f39c12",
            "action": "#27ae60",
            "tool": "#3498db",
        }
        nodes.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "type": node_type,
            "color": color_map.get(node_type, "#95a5a6"),
            "metadata": {k: v for k, v in data.items() if k not in ("label", "type")},
        })

    edges = []
    for src, dst, data in graph.edges(data=True):
        relation = data.get("relation", "RELATED")
        # 不同关系类型的颜色和样式
        edge_style = {
            "CAUSED_BY": {"color": "#e74c3c", "dashes": False},
            "RESOLVED_BY": {"color": "#27ae60", "dashes": False},
            "MAY_TRIGGER": {"color": "#e67e22", "dashes": True},
            "USES": {"color": "#3498db", "dashes": True},
        }
        style = edge_style.get(relation, {"color": "#95a5a6", "dashes": False})
        edges.append({
            "from": src,
            "to": dst,
            "label": relation,
            "color": style["color"],
            "dashes": style["dashes"],
            "reason": data.get("reason", ""),
        })

    return {
        "code": 200,
        "data": {
            "nodes": nodes,
            "edges": edges,
        }
    }
