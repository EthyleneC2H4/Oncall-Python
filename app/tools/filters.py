"""角色级工具过滤 —— 工具可见性的单一事实源

替代 specialists 各自「子串猜工具」（"log" in name.lower()）的散落写法：
每个角色能看到/绑定哪些工具在此集中声明，tests 与运行时共用同一份表。

原则：
- 未知角色默认拒绝（deny-by-default），新增角色必须显式登记
- None 表示不做过滤（全部工具可用，如 ReAct 对话）
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# 角色 → 可用工具名集合；None = 不过滤（全部可用）
ROLE_FILTERS: dict[str, set[str] | None] = {
    # ReAct 对话与规划者：全量工具（规划者只读描述不执行）
    "react_chat": None,
    "planner": None,
    # 多 Agent 并行诊断三专家
    "log_analyst": {
        "search_log",
        "get_topic_info_by_name",
        "describe_topic",
        "list_topics",
        "get_histograms",
    },
    "metric_inspector": {
        "query_cpu_metrics",
        "query_memory_metrics",
    },
    "knowledge_retriever": {
        "retrieve_knowledge",
        "query_alert_graph",
        "predict_alert_cascade",
    },
    # 综合报告者：纯 LLM 综合，不给任何工具
    "synthesizer": set(),
}


def tools_for_role(tools: list[Any], role: str) -> list[Any]:
    """按角色过滤工具列表

    Args:
        tools: langchain 工具对象列表（需有 .name 属性）
        role: 角色名；未在 ROLE_FILTERS 登记的角色返回空列表（deny-by-default）

    Returns:
        该角色可见的工具子集
    """
    allowed = ROLE_FILTERS.get(role)
    if allowed is None:
        if role in ROLE_FILTERS:
            return list(tools)  # 显式登记的不过滤角色
        logger.warning(f"未知角色 {role!r}，按最小权限返回空工具集")
        return []

    filtered = [t for t in tools if getattr(t, "name", "") in allowed]
    dropped = len(tools) - len(filtered)
    if dropped:
        logger.debug(f"角色 {role}: 过滤掉 {dropped} 个越权工具")
    return filtered


def roles_for_tool(tool_name: str) -> list[str]:
    """反向查询：某工具可被哪些（受限）角色使用——测试与文档用途"""
    return [
        role
        for role, allowed in ROLE_FILTERS.items()
        if allowed is not None and tool_name in allowed
    ]


def tools_executable_without_approval(tools: list[Any]) -> list[Any]:
    """剔除需人工确认的高风险工具，返回可被 LLM 自主执行的子集

    用途：mini-ReAct / ToolNode 等绕过 guarded_call 调用点的自主执行池，
    必须先经此过滤——否则确认门只覆盖计划直连路径，LLM 仍可能
    自主调用 restart_instance 类工具。未注册工具不在此拦截（由
    guard 的破坏性启发式与审计兜底）。
    """
    from app.tools.tool_registry import tool_registry

    executable = []
    dropped = 0
    for tool in tools:
        meta = tool_registry.get(getattr(tool, "name", ""))
        if meta is not None and meta.requires_confirmation:
            dropped += 1
            continue
        executable.append(tool)
    if dropped:
        logger.warning(f"已从自主执行池剔除 {dropped} 个需人工确认的高风险工具")
    return executable
