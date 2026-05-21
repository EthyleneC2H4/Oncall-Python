"""知识图谱查询工具 - 从运维知识图谱中查询告警关联信息"""

from langchain_core.tools import tool
from loguru import logger

from app.services.knowledge_graph_service import knowledge_graph_service


@tool
def query_alert_graph(alert_keyword: str) -> str:
    """查询运维知识图谱，获取告警的根因分析、处置建议和级联风险预测。

    当用户询问某类告警的原因、处理方法、可能引发的后续问题时，使用此工具。
    支持中英文关键词，例如：CPU、内存、memory、磁盘、disk、响应慢、服务不可用等。

    Args:
        alert_keyword: 告警相关的关键词，如 "CPU"、"内存"、"磁盘"、"响应慢"、"服务不可用"

    Returns:
        str: 格式化的知识图谱分析结果，包含根因、处置建议和级联风险
    """
    try:
        logger.info(f"知识图谱查询: keyword='{alert_keyword}'")

        context = knowledge_graph_service.format_analysis_context(alert_keyword)
        if not context:
            return f"知识图谱中未找到与 '{alert_keyword}' 相关的告警信息。"

        logger.info(f"知识图谱查询成功，结果长度: {len(context)}")
        return context

    except Exception as e:
        logger.error(f"知识图谱查询失败: {e}")
        return f"知识图谱查询失败: {str(e)}"


@tool
def predict_alert_cascade(alert_keyword: str) -> str:
    """预测告警可能引发的级联告警链路。

    当系统出现某类告警时，使用此工具预测后续可能发生的连锁反应，
    帮助运维人员提前防范。

    Args:
        alert_keyword: 当前告警关键词

    Returns:
        str: 级联告警预测结果
    """
    try:
        logger.info(f"告警级联预测: keyword='{alert_keyword}'")
        prediction = knowledge_graph_service.get_cascade_prediction(alert_keyword)
        logger.info(f"级联预测完成")
        return prediction

    except Exception as e:
        logger.error(f"告警级联预测失败: {e}")
        return f"级联预测失败: {str(e)}"
