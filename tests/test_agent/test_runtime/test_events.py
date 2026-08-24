"""统一事件协议单元测试

覆盖：EventType 语义、AgentEvent 序列化往返、AgentEventEmitter 自增序号。
"""

import pytest
from pydantic import ValidationError

from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType


class TestEventType:
    def test_terminal_types(self):
        """终止事件 = COMPLETE + ERROR，流消费方可据此断开"""
        assert EventType.terminal_types() == {EventType.COMPLETE, EventType.ERROR}

    def test_values_are_stable_strings(self):
        """事件类型值为稳定字符串（SSE 契约只增不改的基础）"""
        assert EventType.TOKEN == "token"
        assert EventType.TOOL_START == "tool_start"
        assert EventType.TOOL_END == "tool_end"
        assert EventType.PLAN_CREATED == "plan_created"

    def test_str_enum_compares_with_plain_string(self):
        """StrEnum 成员可与裸字符串比较（dict.get("type") 场景）"""
        assert EventType.COMPLETE in {"complete", "error"}
        assert EventType.terminal_types() <= {"complete", "error"}


class TestAgentEvent:
    def test_serialization_round_trip(self):
        """model_dump → 反序列化应无损还原"""
        event = AgentEvent(
            seq=1,
            session_id="s1",
            run_id="r1",
            type=EventType.TOOL_START,
            timestamp=1724400000.0,
            payload={"tool": "query_logs", "args": {"kw": "cpu"}},
        )
        dumped = event.model_dump(mode="json")
        restored = AgentEvent(**dumped)

        assert restored == event
        assert restored.type is EventType.TOOL_START
        assert restored.payload["args"] == {"kw": "cpu"}

    def test_payload_defaults_to_empty(self):
        event = AgentEvent(seq=1, session_id="s", run_id="r", type=EventType.COMPLETE, timestamp=0.0)
        assert event.payload == {}

    def test_seq_must_be_positive(self):
        with pytest.raises(ValidationError):
            AgentEvent(seq=0, session_id="s", run_id="r", type=EventType.COMPLETE, timestamp=0.0)


class TestAgentEventEmitter:
    def test_seq_increments_from_one(self):
        emitter = AgentEventEmitter(session_id="s1")
        e1 = emitter.emit(EventType.TOKEN, text="a")
        e2 = emitter.emit(EventType.TOKEN, text="b")

        assert e1.seq == 1
        assert e2.seq == 2
        assert emitter.seq == 2

    def test_session_and_run_id_propagated(self):
        emitter = AgentEventEmitter(session_id="s1", run_id="fixed-run")
        e = emitter.emit(EventType.REPORT, report="# r")

        assert e.session_id == "s1"
        assert e.run_id == "fixed-run"

    def test_run_id_auto_generated_unique(self):
        e1 = AgentEventEmitter(session_id="s").emit(EventType.COMPLETE)
        e2 = AgentEventEmitter(session_id="s").emit(EventType.COMPLETE)

        assert e1.run_id != e2.run_id
        assert e1.run_id  # 非空

    def test_timestamps_monotonic(self):
        emitter = AgentEventEmitter(session_id="s")
        stamps = [emitter.emit(EventType.TOKEN).timestamp for _ in range(5)]
        assert stamps == sorted(stamps)

    def test_payload_kwargs_landed(self):
        e = AgentEventEmitter(session_id="s").emit(
            EventType.STEP_END, current_step="查询日志", steps_done=2, remaining_steps=3
        )
        assert e.payload == {"current_step": "查询日志", "steps_done": 2, "remaining_steps": 3}
