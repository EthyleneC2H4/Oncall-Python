"""工具清单单一事实源测试（P5-d）

收敛回归：诊断四件套此前散落 7 处，任何一处漂移（多/少/改名）
都应在此被钉住。评审加固（#11）：源码级钉住消费方不再自带工具
清单副本——直接引用 app.tools 工具名的模块即回归。
"""

import inspect
from types import SimpleNamespace

from app.agent.runtime.toolsets import knowledge_toolkit, local_tool_map, local_toolkit

EXPECTED_FOUR = {"retrieve_knowledge", "query_alert_graph", "predict_alert_cascade", "get_current_time"}

# 已收敛到 toolsets 的六个消费方（评审 #11 源码钉桩）
_CONSUMER_MODULES = (
    "app.agent.aiops.planner",
    "app.agent.aiops.replanner",
    "app.agent.aiops.executor",
    "app.agent.multi.specialists",
    "app.tools.guard",
    "app.agent.runtime.react_runtime",
)


class TestLocalToolkit:
    def test_contains_exactly_the_four_diagnostic_tools(self):
        names = {t.name for t in local_toolkit()}
        assert names == EXPECTED_FOUR

    def test_fresh_list_each_call(self):
        """返回新列表：调用方拼接/删改不得污染后续调用"""
        pool = local_toolkit()
        pool.clear()
        assert len(local_toolkit()) == 4


class TestKnowledgeToolkit:
    def test_excludes_time_tool(self):
        names = {t.name for t in knowledge_toolkit()}
        assert names == EXPECTED_FOUR - {"get_current_time"}

    def test_is_subset_of_local_pool(self):
        local_names = {t.name for t in local_toolkit()}
        assert {t.name for t in knowledge_toolkit()} <= local_names


class TestLocalToolMap:
    def test_indexed_by_tool_name(self):
        mapping = local_tool_map()
        assert set(mapping) == EXPECTED_FOUR
        for name, tool in mapping.items():
            assert tool.name == name


class TestConsumersUseSingleSource:
    """评审 #11：消费方必须引用 toolsets，不得自带工具名清单"""

    def test_consumer_sources_reference_toolsets_and_name_no_tools(self):
        import importlib

        for module_name in _CONSUMER_MODULES:
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)
            assert (
                "runtime.toolsets" in source or "runtime import toolsets" in source
            ), f"{module_name} 未引用 toolsets 单一事实源"
            assert "get_current_time" not in source.replace(
                "runtime.toolsets", ""
            ), f"{module_name} 源码中出现了具体工具名（应经 toolsets 间接获取）"

    async def test_knowledge_retriever_binds_exactly_knowledge_toolkit(self, monkeypatch):
        """行为钉桩：KnowledgeRetrieverAgent 实际 bind 的工具
        必须与 knowledge_toolkit() 一致（不多不少）"""
        from app.agent.multi.specialists import KnowledgeRetrieverAgent

        bound_names: list[list[str]] = []

        class _FakeLLM:
            async def ainvoke(self, messages):
                return SimpleNamespace(tool_calls=None, content="无工具调用回答")

            def bind_tools(self, tools):
                bound_names.append([t.name for t in tools])
                return self

        class _FakeLLMFactory:
            @staticmethod
            def create_chat_model(**kwargs):
                return _FakeLLM()

        import app.agent.multi.specialists as specialists_module

        monkeypatch.setattr(specialists_module, "LLMFactory", _FakeLLMFactory)

        agent = KnowledgeRetrieverAgent()
        answer = await agent.analyze("CPU 告警")

        assert answer == "无工具调用回答"
        assert agent.confidence == 0.4
        assert len(bound_names) >= 1
        assert set(bound_names[0]) == {t.name for t in knowledge_toolkit()}

    async def test_guard_lookup_path_uses_same_source(self, monkeypatch):
        """guard._find_tool 的本地查找必须命中同一池中的每个名字"""
        from unittest.mock import patch

        from app.tools.guard import _find_tool

        async def no_mcp():
            return []

        with patch("app.agent.mcp_client.get_mcp_tools", no_mcp):
            for name in EXPECTED_FOUR:
                found = await _find_tool(name)
                assert found is not None and found.name == name

        assert await _find_tool("no_such_tool_xyz") is None
