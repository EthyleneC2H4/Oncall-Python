"""guard 执行咽喉测试：权限门 / 参数门 / 确认门 / 审计留痕

覆盖 guarded_call 的四类路径（只读放行、参数拒绝、高风险确认门、
未注册工具防御）与 approve 补执行闭环。
"""

import pytest

import app.tools.guard as guard_module
from app.config import config
from app.services.pending_actions import (
    ActionStatus,
    get_pending_action_store,
    reset_pending_action_store,
)
from app.tools.guard import execute_approved, guarded_call
from app.tools.tool_registry import tool_registry


class FakeTool:
    """最小工具替身：记录 ainvoke 入参并返回固定输出"""

    def __init__(self, name: str, output: str = "工具输出", fail: Exception | None = None):
        self.name = name
        self.output = output
        self.fail = fail
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(dict(args))
        if self.fail is not None:
            raise self.fail
        return self.output


@pytest.fixture
def isolated_guard(tmp_path, monkeypatch):
    """隔离注册表 / 待审动作库 / 审计记录器"""
    saved_registry = dict(tool_registry._registry)

    class Recorder:
        def __init__(self):
            self.entries = []

        def log_tool_call(self, **kwargs):
            self.entries.append(kwargs)

    recorder = Recorder()
    monkeypatch.setattr(guard_module, "audit_logger", recorder)
    monkeypatch.setattr(config, "pending_actions_db_path", str(tmp_path / "actions.db"))
    reset_pending_action_store()
    yield recorder
    reset_pending_action_store()
    tool_registry._registry.clear()
    tool_registry._registry.update(saved_registry)


VALID_SEARCH_ARGS = {"topic_id": "topic-1", "start_time": 1000, "end_time": 2000}


class TestReadOnlyExecution:
    async def test_registered_readonly_executes(self, isolated_guard):
        fake = FakeTool("get_current_time")
        result = await guarded_call(fake, {})

        assert result.ok is True
        assert result.value == "工具输出"
        assert result.needs_confirmation is False
        assert fake.calls == [{}]

    async def test_read_audit_tool_records_audit_entry(self, isolated_guard):
        """search_log 声明 audit=True：执行成功必须留下审计痕迹"""
        fake = FakeTool("search_log")
        result = await guarded_call(fake, dict(VALID_SEARCH_ARGS), request_id="req-1")

        assert result.ok is True
        entries = isolated_guard.entries
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "search_log"
        assert entries[0]["result_status"] == "success"
        assert entries[0]["request_id"] == "req-1"

    async def test_tool_exception_folds_into_result_and_audits(self, isolated_guard):
        """咽喉契约：工具抛异常不外泄，折叠为 ok=False + error 审计"""
        fake = FakeTool("search_log", fail=RuntimeError("连接超时"))
        result = await guarded_call(fake, dict(VALID_SEARCH_ARGS))

        assert result.ok is False
        assert "连接超时" in result.error
        assert isolated_guard.entries[-1]["result_status"] == "error"

    async def test_registered_without_audit_flag_stays_silent(self, isolated_guard):
        """get_current_time 未声明审计：正常执行不留痕（尊重注册表声明）"""
        await guarded_call(FakeTool("get_current_time"), {})
        assert isolated_guard.entries == []


class TestParamGate:
    async def test_missing_required_param_rejected(self, isolated_guard):
        fake = FakeTool("search_log")
        result = await guarded_call(fake, {"start_time": 1, "end_time": 2})

        assert result.ok is False
        assert "参数校验失败" in result.error
        assert "topic_id" in result.error
        assert fake.calls == []  # 参数门拦截，未触达真实工具

    async def test_wrong_param_type_rejected(self, isolated_guard):
        bad = {**VALID_SEARCH_ARGS, "start_time": "not-a-number"}
        result = await guarded_call(FakeTool("search_log"), bad)

        assert result.ok is False
        assert "整数" in result.error

    async def test_out_of_range_param_rejected(self, isolated_guard):
        bad = {**VALID_SEARCH_ARGS, "limit": 5000}  # 上限 1000
        result = await guarded_call(FakeTool("search_log"), bad)

        assert result.ok is False
        assert result.value is None
        assert "limit" in result.error
        assert "1000" in result.error  # 报错携带边界值


class TestConfirmationGate:
    async def test_high_risk_proposes_action_instead_of_executing(self, isolated_guard):
        fake = FakeTool("restart_instance")
        args = {"instance_id": "i-123"}
        result = await guarded_call(fake, args)

        assert result.ok is False
        assert result.needs_confirmation is True
        assert result.action_id
        assert fake.calls == []  # 高风险操作绝不静默执行

        action = get_pending_action_store().get(result.action_id)
        assert action is not None
        assert action.status is ActionStatus.PENDING
        assert action.tool_name == "restart_instance"
        assert action.args == args

    async def test_approve_then_execute_completed_loop(self, isolated_guard, monkeypatch):
        """确认门全流程：propose → 人工批准 → 补执行 → 结果回填"""
        gate_result = await guarded_call(
            FakeTool("restart_instance"), {"instance_id": "i-9"}, session_id="s1"
        )
        store = get_pending_action_store()
        assert store.decide(gate_result.action_id, ActionStatus.APPROVED).status is ActionStatus.APPROVED

        approved_fake = FakeTool("restart_instance", output="重启完成")

        async def fake_find(name):
            return approved_fake if name == "restart_instance" else None

        monkeypatch.setattr(guard_module, "_find_tool", fake_find)

        result = await execute_approved(store.get(gate_result.action_id))

        assert result.ok is True
        assert result.value == "重启完成"
        assert approved_fake.calls == [{"instance_id": "i-9"}]
        refreshed = store.get(gate_result.action_id)
        # 补执行即原子认领 APPROVED→EXECUTED：终态防重放
        assert refreshed.status is ActionStatus.EXECUTED
        assert "重启完成" in refreshed.result_preview
        # 补执行走的是旁路，不应再产生新的待审动作
        assert store.list_pending() == []

    async def test_execute_approved_refuses_non_approved_action(self, isolated_guard):
        """rejected/expired 状态的动作即使被误调也拒绝补执行"""
        gate_result = await guarded_call(FakeTool("restart_instance"), {})
        store = get_pending_action_store()
        store.decide(gate_result.action_id, ActionStatus.REJECTED)

        result = await execute_approved(store.get(gate_result.action_id))
        assert result.ok is False
        assert "approved" in result.error

    async def test_double_decide_is_idempotent(self, isolated_guard):
        action_id = (
            await guarded_call(FakeTool("scale_out"), {})
        ).action_id
        store = get_pending_action_store()

        first = store.decide(action_id, ActionStatus.APPROVED)
        second = store.decide(action_id, ActionStatus.APPROVED)
        assert first.decided_at == second.decided_at  # 未被二次改写

        rejected_attempt = store.decide(action_id, ActionStatus.REJECTED)
        assert rejected_attempt.status is ActionStatus.APPROVED  # 终态不可翻转

    async def test_execute_approved_is_exactly_once(self, isolated_guard, monkeypatch):
        """原子认领：重复补执行第二次必须被拒（防高风险操作重放）"""
        gate_result = await guarded_call(FakeTool("restart_instance"), {"instance_id": "i-7"})
        store = get_pending_action_store()
        store.decide(gate_result.action_id, ActionStatus.APPROVED)

        fake = FakeTool("restart_instance", output="第一次重启")

        async def fake_find(name):
            return fake

        monkeypatch.setattr(guard_module, "_find_tool", fake_find)

        first = await execute_approved(store.get(gate_result.action_id))
        assert first.ok is True
        assert len(fake.calls) == 1

        replay = await execute_approved(store.get(gate_result.action_id))
        assert replay.ok is False
        assert "重放" in replay.error or "approved" in replay.error
        assert len(fake.calls) == 1  # 工具未被二次触达

    async def test_approved_retry_requires_valid_credential(self, isolated_guard):
        """确认门旁路凭证化：伪造/不存在的 action_id 不能跳过确认门"""
        from app.tools.guard import guarded_call as gc

        fake = FakeTool("restart_instance")
        result = await gc(fake, {}, approved_action_id="forged-id")
        assert result.ok is False
        assert "凭证无效" in result.error
        assert fake.calls == []
        assert isolated_guard.entries[-1]["result_status"] == "rejected"

    async def test_approved_retry_credential_must_match_tool(self, isolated_guard):
        """凭证与工具不符同样拒绝（拿 A 动作的批准执行 B 工具）"""
        gate = await guarded_call(FakeTool("restart_instance"), {})
        store = get_pending_action_store()
        store.decide(gate.action_id, ActionStatus.APPROVED)

        other = FakeTool("scale_out")
        result = await guarded_call(other, {}, approved_action_id=gate.action_id)
        assert result.ok is False
        assert "凭证无效" in result.error
        assert other.calls == []

    async def test_destructive_payload_in_nested_values_detected(self, isolated_guard):
        """破坏载荷藏在嵌套 value 里也要拦（曾只查顶层键名）"""
        from app.tools.guard import _looks_destructive

        assert _looks_destructive({"options": {"cmd": "rm -rf /data"}}) is True
        assert _looks_destructive({"sql": ["SELECT 1", "DROP TABLE users"]}) is True
        assert _looks_destructive({"force": True}) is True
        assert _looks_destructive({"query": {"match": "cpu 高"}}) is False
        assert _looks_destructive({"limit": 10}) is False


class TestUnregisteredTools:
    async def test_unregistered_passthrough_executes_and_always_audits(self, isolated_guard):
        """动态 MCP 工具兼容：未注册按只读放行，但强制留痕"""
        fake = FakeTool("mystery_cluster_tool")
        result = await guarded_call(fake, {"cluster": "prod"})

        assert result.ok is True
        entries = isolated_guard.entries
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "mystery_cluster_tool"
        assert entries[0]["result_status"] == "success"

    async def test_unregistered_destructive_args_rejected(self, isolated_guard):
        """防御纵深：未注册却带高危键名的参数形状直接拒绝"""
        fake = FakeTool("mystery_tool")
        result = await guarded_call(fake, {"force": True})

        assert result.ok is False
        assert "破坏性" in result.error
        assert fake.calls == []
        assert isolated_guard.entries[0]["result_status"] == "rejected"

    @pytest.mark.parametrize("risky_key", ["force", "confirm", "drop", "delete_all", "restart", "shutdown"])
    async def test_destructive_heuristic_keys(self, isolated_guard, risky_key):
        from app.tools.guard import _looks_destructive

        assert _looks_destructive({risky_key: 1}) is True
        assert _looks_destructive({risky_key.upper(): 1}) is True  # 大小写不敏感
        assert _looks_destructive({"query": "x", "top_k": 5}) is False
