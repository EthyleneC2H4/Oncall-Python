"""ReAct 运行时 × 长期记忆 集成测试

验证 P2 闭环：轮前召回注入本轮用户消息前缀、轮后写入情景记忆、
memory_enabled=False 时行为与无记忆版本一致（不召回、不写入）。

回归（对抗评审 #16）：输入消息不再携带 SystemMessage —— 系统提示词由
create_agent 统一注入，否则每轮累积进检查点且被最旧优先裁剪误杀最新记忆块。
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.agent.runtime import EventType
from app.agent.runtime.react_runtime import ReActRuntime
from app.services.memory.types import MemoryItem, MemoryType


class AIMessageChunk:
    """与白名单类型名匹配的 token 替身"""

    def __init__(self, text: str):
        self.content_blocks = [{"type": "text", "text": text}]


class RecordingAgent:
    """记录输入并按脚本回放 token 的假 Agent

    流形状对齐真实 langgraph：stream_mode=["messages","updates"] 时
    messages 分支产出 (mode, (消息块, metadata_dict))。
    """

    def __init__(self, chunks: list[AIMessageChunk]):
        self.chunks = chunks
        self.inputs: list[dict[str, Any]] = []

    def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
        async def _gen():
            self.inputs.append(input)
            for chunk in self.chunks:
                yield ("messages", (chunk, {"langgraph_node": "agent"}))

        return _gen()


def make_runtime(monkeypatch, chunks) -> tuple[ReActRuntime, RecordingAgent]:
    runtime = ReActRuntime(tools=[], system_prompt="你是运维助手")
    agent = RecordingAgent(chunks)

    async def fake_ready():
        runtime.agent = agent

    monkeypatch.setattr(runtime, "ensure_ready", fake_ready)
    return runtime, agent


@pytest.fixture
def fake_memory(monkeypatch):
    """替换单例：可配置的假记忆服务"""
    memory = SimpleMemory()
    import app.services.memory as memory_pkg

    monkeypatch.setattr(memory_pkg, "memory_service", memory)
    yield memory


class SimpleMemory:
    def __init__(self):
        self.enabled = True
        self.recall = AsyncMock(return_value=[])
        self.write_episodic = AsyncMock(return_value="mid")

    @staticmethod
    def sample_items() -> list[MemoryItem]:
        return [
            MemoryItem(type=MemoryType.SEMANTIC, content="上次 OOM 根因是内存泄漏", importance=0.8),
            MemoryItem(type=MemoryType.EPISODIC, content="昨夜重启后恢复", importance=0.4),
        ]


class TestMemoryInjection:
    @pytest.mark.asyncio
    async def test_recalled_memory_prefixed_to_user_message(self, monkeypatch, fake_memory):
        fake_memory.recall.return_value = SimpleMemory.sample_items()
        runtime, agent = make_runtime(monkeypatch, [AIMessageChunk("答案")])

        events = [e async for e in runtime.run("内存告警怎么处理", session_id="s1")]

        assert agent.inputs, "Agent 应被调用"
        messages = agent.inputs[0]["messages"]
        # 回归 #16：输入只含本轮用户消息，系统提示词由 create_agent 注入，
        # 不再随轮次累积 SystemMessage 进检查点状态
        assert len(messages) == 1
        human_msg = messages[0]
        assert human_msg.type == "human"
        assert "[相关记忆]" in human_msg.content
        assert "上次 OOM 根因是内存泄漏" in human_msg.content
        # 记忆块为前缀，任务原文保留在后
        assert human_msg.content.endswith("内存告警怎么处理")

        # 召回调用参数正确
        fake_memory.recall.assert_awaited_once()
        assert fake_memory.recall.await_args.args[0] == "内存告警怎么处理"

        # 流正常收尾
        assert events[-1].type is EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_episodic_written_after_turn(self, monkeypatch, fake_memory):
        runtime, _ = make_runtime(monkeypatch, [AIMessageChunk("诊断完成")])

        _ = [e async for e in runtime.run("分析告警", session_id="sess-9")]

        fake_memory.write_episodic.assert_awaited_once()
        args = fake_memory.write_episodic.await_args
        content = args.args[0]
        assert content.startswith("Q: 分析告警")
        assert "诊断完成" in content
        assert args.kwargs["session_id"] == "sess-9"
        assert args.kwargs["metadata"]["runtime"] == "react"

    @pytest.mark.asyncio
    async def test_no_recall_no_write_when_disabled(self, monkeypatch, fake_memory):
        fake_memory.enabled = False
        runtime, agent = make_runtime(monkeypatch, [AIMessageChunk("ok")])

        _ = [e async for e in runtime.run("问题", session_id="s1")]

        fake_memory.recall.assert_not_awaited()
        fake_memory.write_episodic.assert_not_awaited()
        messages = agent.inputs[0]["messages"]
        assert len(messages) == 1
        assert messages[0].content == "问题"  # 无记忆块前缀

    @pytest.mark.asyncio
    async def test_recall_failure_degrades_gracefully(self, monkeypatch, fake_memory):
        """召回抛异常不应影响主流程（失败安全契约）"""
        fake_memory.recall.side_effect = RuntimeError("嵌入服务熔断")
        runtime, agent = make_runtime(monkeypatch, [AIMessageChunk("仍然回答")])

        events = [e async for e in runtime.run("任务", session_id="s1")]

        assert events[-1].type is EventType.COMPLETE
        # 降级为无记忆块的纯任务输入
        assert agent.inputs[0]["messages"][0].content == "任务"

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_create_agent(self, monkeypatch):
        """回归 #16：系统提示词经 create_agent(system_prompt=...) 注入，
        不再以 SystemMessage 形式进入每轮输入"""
        import inspect
        from unittest.mock import patch

        runtime = ReActRuntime(tools=[], system_prompt="你是运维助手")
        captured: dict[str, Any] = {}

        def fake_create_agent(model, tools, **kwargs):  # noqa: ANN001, ARG001
            captured.update(kwargs)

            class _Agent:
                pass

            return _Agent()

        with (
            patch("app.agent.runtime.react_runtime.create_agent", side_effect=fake_create_agent),
            patch(
                "app.agent.runtime.react_runtime.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.agent.runtime.react_runtime.LLMFactory.create_chat_model",
                return_value=object(),
            ),
        ):
            await runtime.ensure_ready()

        assert captured.get("system_prompt") == "你是运维助手"
        # 确认 create_agent 真实签名支持 system_prompt 参数（防止上游变更静默失效）
        from langchain.agents import create_agent as real_create_agent

        assert "system_prompt" in inspect.signature(real_create_agent).parameters

    @pytest.mark.asyncio
    async def test_empty_answer_skips_write(self, monkeypatch, fake_memory):
        """空回答不写情景记忆（避免噪声）"""
        runtime, _ = make_runtime(monkeypatch, [])  # 无任何 token
        _ = [e async for e in runtime.run("任务", session_id="s1")]
        fake_memory.write_episodic.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_importance_scales_with_tool_usage(self, monkeypatch, fake_memory):
        """工具参与度写入情景记忆元数据（重要性加成的依据）"""
        from langchain_core.messages import AIMessage

        runtime, agent = make_runtime(monkeypatch, [AIMessageChunk("有工具参与")])

        def astream_with_tools(input=None, config=None, stream_mode=None):  # noqa: A002
            async def _gen():
                agent.inputs.append(input)
                tool_call_ai = AIMessage(
                    content="",
                    tool_calls=[{"name": "query_alert_graph", "args": {}, "id": "c1"}],
                )
                yield ("updates", {"agent": {"messages": [tool_call_ai]}})
                for chunk in agent.chunks:
                    yield ("messages", (chunk, {"langgraph_node": "agent"}))

            return _gen()

        agent.astream = astream_with_tools  # type: ignore[method-assign]
        events = [e async for e in runtime.run("任务", session_id="s1")]

        assert sum(1 for e in events if e.type is EventType.TOOL_START) == 1
        kwargs = fake_memory.write_episodic.await_args.kwargs
        assert kwargs["metadata"]["tool_calls"] == 1
