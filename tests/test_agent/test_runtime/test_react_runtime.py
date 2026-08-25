"""ReActRuntime 测试

覆盖：双通道（messages/updates）→ TOKEN / TOOL_START / TOOL_END / COMPLETE
事件映射、tool_call id 回填、异常 → ERROR 事件、会话快照/清空、
工作流超时部分回答、检查点 LRU 淘汰、并发首请求单次初始化。
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.runtime.events import EventType
from app.agent.runtime.react_runtime import ReActRuntime


class AIMessageChunk:
    """伪装 AIMessageChunk（type 名需匹配运行时白名单）"""

    def __init__(self, blocks: list | None = None):
        self.content_blocks = blocks


class FakeAgent:
    """按脚本回放 (mode, chunk) 的假 Agent"""

    def __init__(self, script: list[tuple[str, Any]]):
        self._script = script

    def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
        async def _gen():
            for item in self._script:
                yield item

        return _gen()


def make_runtime(script: list[tuple[str, Any]]) -> ReActRuntime:
    rt = ReActRuntime(streaming=True, system_prompt="测试系统提示词")
    rt._initialized = True
    rt.agent = FakeAgent(script)
    return rt


class TestDualChannelStreaming:
    @pytest.mark.asyncio
    async def test_full_event_sequence(self):
        """TOKEN → TOOL_START → TOOL_END → TOKEN → COMPLETE 完整序列"""
        script = [
            # 第一轮模型输出（工具调用意图，无文本）
            ("messages", (AIMessageChunk([{"type": "text", "text": ""}]), {"langgraph_node": "agent"})),
            # agent 节点增量：携带 tool_calls
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": "query_alert_graph", "args": {"kw": "cpu"}, "id": "call_1"}
                                ],
                            )
                        ]
                    }
                },
            ),
            # tools 节点增量：ToolMessage
            (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content="根因: 内存泄漏", name="query_alert_graph", tool_call_id="call_1"
                            )
                        ]
                    }
                },
            ),
            # 第二轮模型流式回答
            ("messages", (AIMessageChunk([{"type": "text", "text": "CPU 告警"}]), {"langgraph_node": "agent"})),
            ("messages", (AIMessageChunk([{"type": "text", "text": "根因是内存泄漏"}]), {"langgraph_node": "agent"})),
        ]

        events = [e async for e in make_runtime(script).run("分析CPU告警", session_id="s1")]

        types = [e.type for e in events]
        assert types == [
            EventType.TOOL_START,
            EventType.TOOL_END,
            EventType.TOKEN,
            EventType.TOKEN,
            EventType.COMPLETE,
        ]

        start_ev = events[0]
        assert start_ev.payload["tool"] == "query_alert_graph"
        assert start_ev.payload["args"] == {"kw": "cpu"}

        end_ev = events[1]
        assert end_ev.payload["tool"] == "query_alert_graph"
        assert end_ev.payload["status"] == "success"
        assert end_ev.payload["result_preview"] == "根因: 内存泄漏"

        assert events[2].payload["text"] == "CPU 告警"
        assert events[2].payload["node"] == "agent"

        complete = events[-1]
        assert complete.payload["answer"] == "CPU 告警根因是内存泄漏"

    @pytest.mark.asyncio
    async def test_empty_text_blocks_skipped(self):
        """空文本块不产生 TOKEN 事件"""
        script = [
            ("messages", (AIMessageChunk([{"type": "text", "text": ""}]), {"langgraph_node": "agent"})),
            ("messages", (AIMessageChunk(None), {"langgraph_node": "agent"})),
        ]

        events = [e async for e in make_runtime(script).run("问题", session_id="s")]

        assert [e.type for e in events] == [EventType.COMPLETE]

    @pytest.mark.asyncio
    async def test_non_ai_message_tokens_ignored(self):
        """非 AIMessageChunk 类型的消息通道数据被忽略"""
        class HumanChunk:
            pass

        script = [
            ("messages", (HumanChunk(), {"langgraph_node": "agent"})),
        ]

        events = [e async for e in make_runtime(script).run("问题", session_id="s")]

        assert [e.type for e in events] == [EventType.COMPLETE]

    @pytest.mark.asyncio
    async def test_tool_end_without_start_falls_back_to_message_name(self):
        """TOOL_START 缺失时 TOOL_END 使用 ToolMessage.name"""
        script = [
            (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(content="ok", name="query_logs", tool_call_id="orphan")
                        ]
                    }
                },
            )
        ]

        events = [e async for e in make_runtime(script).run("q", session_id="s")]

        assert len(events) == 2
        assert events[0].type is EventType.TOOL_END
        assert events[0].payload["tool"] == "query_logs"

    @pytest.mark.asyncio
    async def test_error_tool_message_marks_error_status(self):
        tool_msg = ToolMessage(
            content="连接超时", name="query_logs", tool_call_id="c1", status="error"
        )
        script = [("updates", {"tools": {"messages": [tool_msg]}})]

        events = [e async for e in make_runtime(script).run("q", session_id="s")]

        assert events[0].payload["status"] == "error"

    @pytest.mark.asyncio
    async def test_exception_yields_terminal_error_event(self):
        """异常转换为 ERROR 终止事件，流正常收尾（不向消费方抛异常）"""
        class ExplodingAgent:
            def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
                async def _gen():
                    raise RuntimeError("OpenRouter 429")
                    yield None  # pragma: no cover - 使其成为生成器

                return _gen()

        rt = ReActRuntime()
        rt._initialized = True
        rt.agent = ExplodingAgent()

        events = [e async for e in rt.run("q", session_id="s")]

        assert [e.type for e in events] == [EventType.ERROR]
        assert "OpenRouter 429" in events[0].payload["message"]


class TestSessionLifecycle:
    def test_reset_delegates_to_checkpointer(self):
        from unittest.mock import MagicMock

        rt = ReActRuntime()
        rt.checkpointer = MagicMock()

        assert rt.reset("sess-9") is True
        rt.checkpointer.delete_thread.assert_called_once_with("sess-9")

    def test_reset_failure_returns_false(self):
        from unittest.mock import MagicMock

        rt = ReActRuntime()
        rt.checkpointer = MagicMock()
        rt.checkpointer.delete_thread.side_effect = RuntimeError("boom")

        assert rt.reset("sess-9") is False

    def test_snapshot_extracts_history(self):
        """快照从检查点提取 user/assistant 历史（跳过 SystemMessage）"""
        from datetime import datetime
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # CheckpointTuple 元组形态：首元素为 checkpoint dict
        checkpoint = {
            "channel_values": {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(content="你好"),
                    AIMessage(content="你好，我是运维助手"),
                ]
            }
        }

        rt = ReActRuntime()
        rt.checkpointer = MagicMock()
        rt.checkpointer.get.return_value = (checkpoint, "metadata")

        snap = rt.snapshot("sess-1")
        history = snap["messages"]

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"
        # 时间戳自动补齐为 ISO 字符串
        datetime.fromisoformat(history[0]["timestamp"])

    def test_snapshot_no_checkpoint_returns_empty(self):
        from unittest.mock import MagicMock

        rt = ReActRuntime()
        rt.checkpointer = MagicMock()
        rt.checkpointer.get.return_value = None

        assert rt.snapshot("missing") == {"messages": []}


class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_ensure_ready_loads_mcp_tools_once(self, monkeypatch):
        """MCP 工具经 get_mcp_tools 装载一次；重复调用幂等"""
        calls = []

        async def fake_get_mcp_tools(ttl_seconds=30.0, *, refresh=False):
            calls.append(refresh)
            return [object()]

        monkeypatch.setattr("app.agent.runtime.react_runtime.get_mcp_tools", fake_get_mcp_tools)

        captured = {}

        def fake_create_agent(model, tools, checkpointer=None, **kwargs):
            captured["tools"] = tools
            return object()

        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.create_agent", fake_create_agent
        )
        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.LLMFactory",
            type(
                "F",
                (),
                {
                    "create_chat_model": staticmethod(
                        lambda model=None, temperature=0.7, streaming=True: object()
                    )
                },
            ),
        )

        rt = ReActRuntime(system_prompt="p")
        await rt.ensure_ready()
        await rt.ensure_ready()  # 第二次应直接返回

        assert calls == [False]  # 只装载一次
        assert len(captured["tools"]) == 5  # 默认四件套 + 1 个 MCP 工具

    @pytest.mark.asyncio
    async def test_concurrent_first_requests_initialize_once(self, monkeypatch):
        """并发首请求：锁内二次检查，MCP 拉取与图编译各只发生一次"""
        mcp_calls: list[bool] = []
        builds = {"count": 0}

        async def fake_get_mcp_tools(ttl_seconds=30.0, *, refresh=False):
            await asyncio.sleep(0.01)  # 放大竞态窗口
            mcp_calls.append(True)
            return [object()]

        def fake_create_agent(model, tools, checkpointer=None, **kwargs):
            builds["count"] += 1
            return object()

        monkeypatch.setattr("app.agent.runtime.react_runtime.get_mcp_tools", fake_get_mcp_tools)
        monkeypatch.setattr("app.agent.runtime.react_runtime.create_agent", fake_create_agent)
        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.LLMFactory",
            type(
                "F",
                (),
                {
                    "create_chat_model": staticmethod(
                        lambda model=None, temperature=0.7, streaming=True: object()
                    )
                },
            ),
        )

        rt = ReActRuntime(system_prompt="p")
        await asyncio.gather(rt.ensure_ready(), rt.ensure_ready(), rt.ensure_ready())

        assert len(mcp_calls) == 1
        assert builds["count"] == 1
        assert rt._initialized is True


class TestWorkflowTimeout:
    @pytest.mark.asyncio
    async def test_timeout_yields_partial_answer_complete(self, monkeypatch):
        """整体 deadline 超时 → 已产出的部分回答以 timed_out=True 的 COMPLETE 收尾"""

        class HangingAgent:
            @staticmethod
            def astream(input=None, config=None, stream_mode=None):  # noqa: A002
                async def _gen():
                    yield (
                        "messages",
                        (AIMessageChunk([{"type": "text", "text": "部分回答"}]), {"langgraph_node": "agent"}),
                    )
                    while True:
                        await asyncio.sleep(3600)  # 停滞：模拟 OpenRouter 挂死

                return _gen()

        # 先建实例（__init__ 读真实 config.rag_model），再注入短超时配置
        rt = make_runtime([])
        rt.agent = HangingAgent()
        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.config",
            SimpleNamespace(workflow_timeout_seconds=0.05, checkpoint_max_threads=50),
        )

        events = [e async for e in rt.run("慢任务", session_id="timeout-test")]

        types = [e.type for e in events]
        assert EventType.TOKEN in types  # 部分回答已先行流出
        assert types[-1] is EventType.COMPLETE  # 终止事件是 COMPLETE 不是 ERROR
        payload = events[-1].payload
        assert payload["timed_out"] is True
        assert "部分回答" in payload["answer"]
        assert "超时" in payload["message"]

        # 清理：HangingAgent 的 sleep 循环随超时取消，无需额外处理


class TestCheckpointLru:
    def _make_lru_runtime(self) -> ReActRuntime:
        rt = ReActRuntime.__new__(ReActRuntime)  # 只需 LRU 相关属性
        rt.checkpointer = MagicMock()
        rt._thread_last_access = {}
        return rt

    def _patch_config(self, monkeypatch, max_threads: int) -> None:
        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.config",
            SimpleNamespace(checkpoint_max_threads=max_threads, workflow_timeout_seconds=30),
        )

    def test_no_eviction_under_limit(self, monkeypatch):
        self._patch_config(monkeypatch, max_threads=3)
        rt = self._make_lru_runtime()

        for tid in ("a", "b"):
            rt._touch_thread(tid)

        assert set(rt._thread_last_access) == {"a", "b"}
        rt.checkpointer.delete_thread.assert_not_called()

    def test_oldest_thread_evicted_over_limit(self, monkeypatch):
        self._patch_config(monkeypatch, max_threads=2)
        rt = self._make_lru_runtime()

        rt._touch_thread("a")
        time.sleep(0.005)
        rt._touch_thread("b")
        time.sleep(0.005)
        rt._touch_thread("c")  # 超 2：淘汰最久未活跃的 a

        assert set(rt._thread_last_access) == {"b", "c"}
        rt.checkpointer.delete_thread.assert_called_once_with("a")

    def test_recently_revisited_thread_survives(self, monkeypatch):
        self._patch_config(monkeypatch, max_threads=2)
        rt = self._make_lru_runtime()

        rt._touch_thread("a")
        time.sleep(0.005)
        rt._touch_thread("b")
        time.sleep(0.005)
        rt._touch_thread("a")  # a 重获活跃 → b 成为最久未活跃
        time.sleep(0.005)
        rt._touch_thread("c")

        assert set(rt._thread_last_access) == {"a", "c"}
        rt.checkpointer.delete_thread.assert_called_once_with("b")

    def test_eviction_failure_swallowed(self, monkeypatch):
        self._patch_config(monkeypatch, max_threads=1)
        rt = self._make_lru_runtime()
        rt.checkpointer.delete_thread.side_effect = RuntimeError("delete 失败")

        rt._touch_thread("a")
        rt._touch_thread("b")  # 淘汰失败不影响主流程

        assert set(rt._thread_last_access) == {"b"}
