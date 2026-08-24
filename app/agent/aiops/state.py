"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现，增加诊断轨迹追踪和降级状态
"""

import operator
from typing import Annotated, TypedDict


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表，旧 List[str] 契约）
    plan: list[str]

    # 结构化计划（P3）：与 plan 一一对应的步骤字典列表
    # 每项: {"id","description","tool","args","depends_on","expected_evidence"}
    plan_structured: list[dict]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[list[tuple], operator.add]

    # 最终响应/报告
    response: str

    # 知识图谱上下文（由 planner 注入）
    kg_context: str

    # 查询意图（由 planner 路由分类）
    query_intent: str

    # 诊断事件流（推理轨迹），使用 operator.add 追加
    diagnosis_events: Annotated[list[dict], operator.add]

    # 错误上下文（降级策略追踪），使用 operator.add 追加
    # 每条: {"step": str, "error_type": str, "error_msg": str}
    error_context: Annotated[list[dict], operator.add]

    # 当前降级等级
    degradation_level: str
