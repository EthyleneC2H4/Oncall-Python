"""executor 观测 ID 贯通测试：state 里的 session/request ID 必须到达 guarded_call

评审修复回归：生产痕迹曾恒为空 ID，BFCL 回放的会话过滤因此失效。
"""

from unittest.mock import patch

from app.agent.aiops.executor import executor
from app.agent.aiops.state import PlanExecuteState
from app.tools.guard import GuardResult


def _state(**overrides) -> PlanExecuteState:
    defaults: dict = {
        "input": "诊断",
        "plan": ["查知识库"],
        "plan_structured": [
            {"id": "s0", "description": "查知识库",
             "tool": "retrieve_knowledge", "args": {"query": "cpu 高"},
             "depends_on": [], "expected_evidence": ""}
        ],
        "past_steps": [],
        "response": "",
        "kg_context": "",
        "query_intent": "DIAGNOSTIC",
        "diagnosis_events": [],
        "error_context": [],
        "degradation_level": "none",
    }
    defaults.update(overrides)
    return PlanExecuteState(**defaults)


class TestTraceIdThreading:
    async def test_guarded_call_receives_state_ids(self):
        captured: dict = {}

        async def fake_guarded_call(tool, args, *, request_id="", session_id=""):
            captured["request_id"] = request_id
            captured["session_id"] = session_id
            return GuardResult(ok=True, value="ok")

        with (
            patch("app.tools.guard.guarded_call", fake_guarded_call),
            patch("app.agent.aiops.executor.get_mcp_tools", return_value=[]),
        ):
            await executor(_state(session_id="sess-42", request_id="req-7"))

        assert captured == {"request_id": "req-7", "session_id": "sess-42"}

    async def test_missing_ids_default_to_empty(self):
        """旧构造点不带 ID 字段（NotRequired）→ 空串兜底而非 KeyError"""
        captured: dict = {}

        async def fake_guarded_call(tool, args, *, request_id="", session_id=""):
            captured["ids"] = (request_id, session_id)
            return GuardResult(ok=True, value="ok")

        state = _state()
        assert "session_id" not in state  # NotRequired 字段确实可缺省

        with (
            patch("app.tools.guard.guarded_call", fake_guarded_call),
            patch("app.agent.aiops.executor.get_mcp_tools", return_value=[]),
        ):
            await executor(state)

        assert captured["ids"] == ("", "")

    async def test_runtime_injects_session_into_initial_state(self):
        """runtime 层注入：initial_state 携带 session_id"""
        import inspect

        from app.agent.runtime.plan_execute_runtime import PlanExecuteRuntime

        source = inspect.getsource(PlanExecuteRuntime.run)
        assert '"session_id": session_id' in source
