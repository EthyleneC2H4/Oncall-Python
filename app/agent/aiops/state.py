"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现，增加诊断轨迹追踪
"""

from typing import List, TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    response: str

    # 知识图谱上下文（由 planner 注入）
    kg_context: str

    # 查询意图（由 planner 路由分类）
    query_intent: str

    # 诊断事件流（推理轨迹），使用 operator.add 追加
    diagnosis_events: Annotated[List[dict], operator.add]
