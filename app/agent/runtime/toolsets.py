"""本地运维工具清单的单一事实源（P5）

此前「诊断四件套」在 executor（直连/ReAct 两处）、planner、replanner、
guard 按名查找、react_runtime 默认池共 7 处各写一遍——新增或下线一个
本地工具要同步改 7 个文件，漏一处即出现「计划绑定了不存在的工具」。
收敛后只有这里枚举一次；specialists 的知识三件套是它的角色子集。

注意：本模块只做清单聚合，不做权限/确认判断（那是 guard 与
filters 的职责）。
"""

from typing import Any

from app.tools import (
    get_current_time,
    predict_alert_cascade,
    query_alert_graph,
    retrieve_knowledge,
)


def local_toolkit() -> list[Any]:
    """本地诊断四件套（每次返回新列表，调用方可自由拼接 MCP 工具）"""
    return [
        retrieve_knowledge,
        query_alert_graph,
        predict_alert_cascade,
        get_current_time,
    ]


def knowledge_toolkit() -> list[Any]:
    """知识诊断三件套：specialists 候选池（时间工具与告警诊断无关）"""
    return [t for t in local_toolkit() if t.name != get_current_time.name]


def local_tool_map() -> dict[str, Any]:
    """按工具名索引的本地池（guard 补执行路径查找用）"""
    return {t.name: t for t in local_toolkit()}
