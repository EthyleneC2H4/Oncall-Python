"""角色级工具过滤矩阵测试 —— ROLE_FILTERS 单一事实源"""

import pytest

from app.tools.filters import ROLE_FILTERS, roles_for_tool, tools_for_role


class DummyTool:
    def __init__(self, name: str):
        self.name = name


ALL_TOOLS = [
    "search_log",
    "get_topic_info_by_name",
    "describe_topic",
    "list_topics",
    "get_histograms",
    "query_cpu_metrics",
    "query_memory_metrics",
    "retrieve_knowledge",
    "query_alert_graph",
    "predict_alert_cascade",
    "get_current_time",
]
TOOLS = [DummyTool(n) for n in ALL_TOOLS]


def _names(tools) -> set[str]:
    return {t.name for t in tools}


class TestRoleMatrix:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("log_analyst",
             {"search_log", "get_topic_info_by_name", "describe_topic",
              "list_topics", "get_histograms"}),
            ("metric_inspector", {"query_cpu_metrics", "query_memory_metrics"}),
            ("knowledge_retriever",
             {"retrieve_knowledge", "query_alert_graph", "predict_alert_cascade"}),
            ("synthesizer", set()),
        ],
    )
    def test_specialist_roles_exact_sets(self, role, expected):
        assert _names(tools_for_role(TOOLS, role)) == expected

    @pytest.mark.parametrize("role", ["react_chat", "planner"])
    def test_unfiltered_roles_get_everything(self, role):
        assert _names(tools_for_role(TOOLS, role)) == set(ALL_TOOLS)

    def test_unknown_role_denied_by_default(self):
        assert tools_for_role(TOOLS, "hacker_role") == []
        assert tools_for_role(TOOLS, "") == []


class TestRegistryConsistency:
    def test_filter_names_reference_real_tools(self):
        """过滤集合里的名字必须真实存在于注册表或本地工具池（防拼写漂移）

        参照系取自产品代码而非测试常量，避免「两张表互相印证」的假绿。
        """
        from app.tools import (
            get_current_time,
            predict_alert_cascade,
            query_alert_graph,
            retrieve_knowledge,
        )
        from app.tools.tool_registry import tool_registry

        known = set(tool_registry._registry.keys()) | {
            get_current_time.name,
            retrieve_knowledge.name,
            query_alert_graph.name,
            predict_alert_cascade.name,
        }
        for role, allowed in ROLE_FILTERS.items():
            if allowed is None:
                continue
            unknown = allowed - known
            assert not unknown, f"角色 {role} 引用了未知工具: {unknown}"

    def test_specialist_roles_get_nonempty_from_registry_pool(self):
        """三专家对注册表工具池过滤后不得为空（防 deny-by-default 误伤）"""
        from app.tools.tool_registry import tool_registry

        class _Named:
            def __init__(self, name):
                self.name = name

        pool = [_Named(n) for n in tool_registry._registry.keys()]
        for role in ("log_analyst", "metric_inspector", "knowledge_retriever"):
            got = tools_for_role(pool, role)
            assert got, f"角色 {role} 对注册表工具池过滤后为空"
            assert {t.name for t in got} <= ROLE_FILTERS[role]

    def test_roles_for_tool_reverse_lookup(self):
        assert roles_for_tool("retrieve_knowledge") == ["knowledge_retriever"]
        assert roles_for_tool("query_cpu_metrics") == ["metric_inspector"]
        assert roles_for_tool("get_current_time") == []
        assert set(roles_for_tool("search_log")) == {"log_analyst"}

    def test_synthesizer_has_zero_tools_by_design(self):
        """综合者纯 LLM 归纳，不给工具是刻意的最小权限声明"""
        assert ROLE_FILTERS["synthesizer"] == set()
