"""PlanExecuteRuntime 流式行为测试

核心回归：假流式修复 —— 旧实现在 wait_for 内先消费完整 graph 流再逐个 yield，
前端在整个工作流结束前收不到任何事件。本测试用可控制节奏的 FakeGraph 验证
「首事件先于流结束到达」。

另覆盖：节点增量 → 事件映射（translate_graph_update）、超时降级路径、
snapshot / reset。
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.runtime.events import AgentEventEmitter, EventType
from app.agent.runtime.plan_execute_runtime import (
    NODE_EXECUTOR,
    NODE_PLANNER,
    NODE_REPLANNER,
    PlanExecuteRuntime,
    translate_graph_update,
)

# ──────────────── 测试替身 ────────────────


class FakeGraph:
    """按脚本回放节点增量的假图，支持事件门控与延迟"""

    def __init__(self, script: list[dict[str, Any]], final_values: dict | None = None):
        self._script = script
        self._final_values = final_values if final_values is not None else {"response": "最终报告"}

    def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
        async def _gen():
            for delay, update in self._script:
                await asyncio.sleep(delay)
                yield update
            self.stream_finished_at = time.monotonic()

        return _gen()

    def get_state(self, config=None):
        return SimpleNamespace(values=dict(self._final_values))


class TestTranslateGraphUpdate:
    """节点状态增量 → 统一事件的映射规则"""

    def setup_emitter(self):
        return AgentEventEmitter(session_id="s1")

    def test_planner_full_payload(self):
        emitter = self.setup_emitter()
        events = translate_graph_update(
            {
                NODE_PLANNER: {
                    "plan": ["步骤1", "步骤2"],
                    "kg_context": "CPU告警: 内存泄漏",
                    "query_intent": "DIAGNOSTIC",
                    "diagnosis_events": [{"event_type": "routing"}],
                }
            },
            emitter,
        )

        assert len(events) == 1
        ev = events[0]
        assert ev.type is EventType.PLAN_CREATED
        assert ev.payload["plan"] == ["步骤1", "步骤2"]
        assert ev.payload["kg_context"] == "CPU告警: 内存泄漏"
        assert ev.payload["query_intent"] == "DIAGNOSTIC"
        assert "共 2 个步骤" in ev.payload["message"]

    def test_planner_empty_optional_fields_omitted(self):
        """空上下文字段不应出现在 payload（对齐旧 _format_planner_event）"""
        emitter = self.setup_emitter()
        events = translate_graph_update({NODE_PLANNER: {"plan": ["a"], "kg_context": ""}}, emitter)

        assert events[0].payload.get("kg_context") is None
        assert events[0].payload.get("diagnosis_events") is None

    def test_executor_step_end(self):
        emitter = self.setup_emitter()
        events = translate_graph_update(
            {
                NODE_EXECUTOR: {
                    "plan": ["剩余步骤"],
                    "past_steps": [("查询日志", "发现 OOM 异常堆栈" + "x" * 500)],
                }
            },
            emitter,
        )

        assert len(events) == 1
        ev = events[0]
        assert ev.type is EventType.STEP_END
        assert ev.payload["current_step"] == "查询日志"
        # 结果预览截断到 300 字符
        assert len(ev.payload["result_preview"]) == 300
        assert ev.payload["steps_done"] == 1
        assert ev.payload["remaining_steps"] == 1

    def test_executor_no_past_steps_is_step_start(self):
        emitter = self.setup_emitter()
        events = translate_graph_update({NODE_EXECUTOR: {"plan": [], "past_steps": []}}, emitter)

        assert events[0].type is EventType.STEP_START

    def test_replanner_with_response_is_report(self):
        emitter = self.setup_emitter()
        events = translate_graph_update(
            {NODE_REPLANNER: {"response": "# 报告内容", "plan": []}}, emitter
        )

        assert events[0].type is EventType.REPORT
        assert events[0].payload["report"] == "# 报告内容"

    def test_replanner_continue_is_replan(self):
        emitter = self.setup_emitter()
        events = translate_graph_update(
            {NODE_REPLANNER: {"response": "", "plan": ["下一步"]}}, emitter
        )

        assert events[0].type is EventType.REPLAN
        assert events[0].payload["remaining_steps"] == 1

    def test_unknown_node_ignored(self):
        emitter = self.setup_emitter()
        events = translate_graph_update({"__meta__": {}}, emitter)
        assert events == []


# ──────────────── 假流式回归 ────────────────


@pytest.fixture
def runtime_with_fast_timeout(monkeypatch):
    """注入短整体超时，避免真实 180s 配置影响测试"""
    monkeypatch.setattr(
        "app.agent.runtime.plan_execute_runtime.config",
        SimpleNamespace(workflow_timeout_seconds=5.0),
    )

    def _make(graph):
        rt = PlanExecuteRuntime.__new__(PlanExecuteRuntime)  # 跳过真实图构建
        rt.checkpointer = None
        rt.graph = graph
        rt._session_locks = {}  # 会话互斥锁表（__new__ 绕过了 __init__）
        return rt

    return _make


class TestTrueStreaming:
    @pytest.mark.asyncio
    async def test_first_event_arrives_before_stream_ends(self, runtime_with_fast_timeout):
        """假流式回归：首个事件必须在流结束前到达消费方

        FakeGraph 先立刻产出 planner 增量，停顿 0.3s 后再产出后续增量。
        若实现退化为「攒齐再吐」，所有事件的接收时间都会贴近流结束时间，
        本断言将以毫秒级差距失败。
        """
        received: list[tuple[float, Any]] = []

        def on_event(ev):
            received.append((time.monotonic(), ev))

        script = [
            (0.0, {NODE_PLANNER: {"plan": ["步骤1", "步骤2"]}}),
            (0.3, {NODE_EXECUTOR: {"plan": [], "past_steps": [("步骤1", "完成")]}}),
            (0.0, {NODE_REPLANNER: {"response": "# 报告", "plan": []}}),
        ]

        class TimedFakeGraph(FakeGraph):
            stream_finished_at = float("inf")

        graph = TimedFakeGraph(script)
        runtime = runtime_with_fast_timeout(graph)

        async for event in runtime.run("诊断任务", session_id="stream-test"):
            on_event(event)

        assert received, "应至少收到一个事件"
        first_received_at = received[0][0]
        stream_finished_at = graph.stream_finished_at

        # 首事件到达时流尚未结束（留 0.15s 余量对抗调度抖动）
        assert first_received_at < stream_finished_at - 0.15, (
            "首个事件应在流结束前 ≥0.15s 到达；若失败说明退化为假流式"
        )
        # 事件类型序列符合预期
        types = [ev.type for _, ev in received]
        assert types == [
            EventType.PLAN_CREATED,
            EventType.STEP_END,
            EventType.REPORT,
            EventType.COMPLETE,
        ]

    @pytest.mark.asyncio
    async def test_complete_carries_final_response(self, runtime_with_fast_timeout):
        graph = FakeGraph(
            [(0.0, {NODE_REPLANNER: {"response": "# 最终诊断报告", "plan": []}})],
            final_values={"response": "# 最终诊断报告"},
        )
        runtime = runtime_with_fast_timeout(graph)

        terminal = [e async for e in runtime.run("任务", session_id="t1") if e.type is EventType.COMPLETE]

        assert len(terminal) == 1
        payload = terminal[0].payload
        assert payload["response"] == "# 最终诊断报告"
        assert payload["timed_out"] is False
        assert payload["message"] == "任务执行完成"


class TestWorkflowTimeout:
    @pytest.mark.asyncio
    async def test_timeout_produces_partial_report(self, monkeypatch):
        """整体超时应产出 timed_out=True 的 COMPLETE 与兜底文案"""
        monkeypatch.setattr(
            "app.agent.runtime.plan_execute_runtime.config",
            SimpleNamespace(workflow_timeout_seconds=0.05),
        )

        class SlowGraph(FakeGraph):
            def __init__(self):
                super().__init__([{NODE_PLANNER: {"plan": ["慢步骤"]}}])
                self._final_values = {}  # 无 response → 触发兜底文案

            def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
                async def _gen():
                    while True:
                        await asyncio.sleep(1)  # 永不完成
                        yield {}

                return _gen()

        rt = PlanExecuteRuntime.__new__(PlanExecuteRuntime)
        rt.checkpointer = None
        rt.graph = SlowGraph()
        rt._session_locks = {}  # 会话互斥锁表（__new__ 绕过了 __init__）

        events = [e async for e in rt.run("慢任务", session_id="timeout-test")]

        complete_events = [e for e in events if e.type is EventType.COMPLETE]
        assert len(complete_events) == 1
        payload = complete_events[0].payload
        assert payload["timed_out"] is True
        assert "诊断超时" in payload["response"]
        assert payload["message"] == "任务超时，已生成部分报告"

    @pytest.mark.asyncio
    async def test_exception_yields_error_event(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent.runtime.plan_execute_runtime.config",
            SimpleNamespace(workflow_timeout_seconds=5.0),
        )

        class ExplodingGraph(FakeGraph):
            def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
                async def _gen():
                    raise RuntimeError("MCP 连接失败")
                    yield {}  # pragma: no cover - 使其成为生成器

                return _gen()

        rt = PlanExecuteRuntime.__new__(PlanExecuteRuntime)
        rt.checkpointer = None
        rt.graph = ExplodingGraph([])
        rt._session_locks = {}  # 会话互斥锁表（__new__ 绕过了 __init__）

        events = [e async for e in rt.run("任务", session_id="err-test")]

        error_events = [e for e in events if e.type is EventType.ERROR]
        assert len(error_events) == 1
        assert "MCP 连接失败" in error_events[0].payload["message"]


class TestSessionLifecycle:
    def test_snapshot_and_reset_delegate_to_checkpointer(self):
        from unittest.mock import MagicMock

        rt = PlanExecuteRuntime.__new__(PlanExecuteRuntime)
        rt.checkpointer = MagicMock()
        rt.checkpointer.delete_thread.return_value = None
        # snapshot 需要 graph.get_state
        rt.graph = MagicMock()
        rt.graph.get_state.return_value = SimpleNamespace(values={"plan": ["x"], "input": "任务"})

        snap = rt.snapshot("sess-1")
        assert snap == {"values": {"plan": ["x"], "input": "任务"}}

        assert rt.reset("sess-1") is True
        rt.checkpointer.delete_thread.assert_called_once_with("sess-1")
