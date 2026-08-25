"""待审动作存储测试：状态机 / TTL 惰性过期 / 容错读取"""

import json
import time

import pytest

from app.services.pending_actions import ActionStatus, PendingActionStore


@pytest.fixture
def store(tmp_path):
    s = PendingActionStore(db_path=str(tmp_path / "pa.db"), ttl_seconds=900.0)
    yield s
    s.close()


def _backdate(s: PendingActionStore, action_id: str, seconds: float) -> None:
    """把记录的 created_at 回拨（模拟陈旧 pending）"""
    with s._lock:
        s._conn.execute(
            "UPDATE pending_actions SET created_at = ? WHERE action_id = ?",
            (time.time() - seconds, action_id),
        )
        s._conn.commit()


class TestProposeAndGet:
    def test_roundtrip_preserves_fields(self, store):
        action = store.propose(
            tool_name="restart_instance",
            args={"instance_id": "i-1"},
            reason="高风险",
            session_id="sess-1",
            request_id="req-1",
        )

        assert action.status is ActionStatus.PENDING
        loaded = store.get(action.action_id)
        assert loaded is not None
        assert loaded.tool_name == "restart_instance"
        assert loaded.args == {"instance_id": "i-1"}
        assert loaded.reason == "高风险"
        assert loaded.session_id == "sess-1"
        assert loaded.request_id == "req-1"

    def test_get_unknown_returns_none(self, store):
        assert store.get("no-such-id") is None


class TestDecideStateMachine:
    def test_approve_transition_sets_decided_at(self, store):
        action = store.propose(tool_name="scale_out")
        decided = store.decide(action.action_id, ActionStatus.APPROVED)

        assert decided.status is ActionStatus.APPROVED
        assert decided.decided_at is not None

    def test_reject_transition(self, store):
        action = store.propose(tool_name="scale_out")
        assert store.decide(action.action_id, ActionStatus.REJECTED).status is ActionStatus.REJECTED

    def test_double_decide_short_circuits(self, store):
        """已裁决动作不可被二次改写：幂等返回现状态"""
        action = store.propose(tool_name="scale_out")
        first = store.decide(action.action_id, ActionStatus.APPROVED)

        second = store.decide(action.action_id, ActionStatus.APPROVED)
        assert second.decided_at == first.decided_at
        assert store.decide(action.action_id, ActionStatus.REJECTED).status is ActionStatus.APPROVED

    def test_illegal_decision_raises(self, store):
        action = store.propose(tool_name="scale_out")
        with pytest.raises(ValueError):
            store.decide(action.action_id, ActionStatus.EXPIRED)


class TestTtlExpiry:
    def test_stale_pending_lazily_marked_expired_on_read(self, store):
        action = store.propose(tool_name="restart_instance")
        _backdate(store, action.action_id, seconds=store.ttl_seconds + 60)

        loaded = store.get(action.action_id)
        assert loaded.status is ActionStatus.EXPIRED

    def test_fresh_pending_survives(self, store):
        action = store.propose(tool_name="restart_instance")
        assert store.get(action.action_id).status is ActionStatus.PENDING

    def test_list_pending_excludes_expired_by_default(self, store):
        fresh = store.propose(tool_name="a_tool")
        stale = store.propose(tool_name="b_tool")
        _backdate(store, stale.action_id, seconds=10_000)

        pending_ids = [a.action_id for a in store.list_pending()]
        assert fresh.action_id in pending_ids
        assert stale.action_id not in pending_ids
        assert stale.action_id in [a.action_id for a in store.list_pending(include_expired=True)]

    def test_list_pending_newest_first(self, store):
        old = store.propose(tool_name="a_tool")
        new = store.propose(tool_name="b_tool")
        _backdate(store, old.action_id, seconds=5)

        ids = [a.action_id for a in store.list_pending()]
        assert ids.index(new.action_id) < ids.index(old.action_id)


class TestTolerance:
    def test_attach_result_truncates_to_500_chars(self, store):
        action = store.propose(tool_name="t")
        store.attach_result(action.action_id, "长" * 2000)

        assert len(store.get(action.action_id).result_preview) == 500

    def test_corrupt_args_json_degrades_to_empty_dict(self, store):
        """手工写入损坏行：get 不炸，args 退化为 {}"""
        with store._lock:
            store._conn.execute(
                "INSERT INTO pending_actions (action_id, tool_name, args, created_at)"
                " VALUES (?, ?, ?, ?)",
                ("corrupt1", "t", "{not-json", time.time()),
            )
            store._conn.commit()

        action = store.get("corrupt1")
        assert action is not None
        assert action.args == {}

    def test_corrupt_status_degrades_to_pending(self, store):
        with store._lock:
            store._conn.execute(
                "INSERT INTO pending_actions (action_id, tool_name, args, status, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                ("corrupt2", "t", json.dumps({}), "weird-status", time.time()),
            )
            store._conn.commit()

        assert store.get("corrupt2").status is ActionStatus.PENDING


class TestAtomicTransition:
    """transition()：条件 UPDATE 单语句认领——并发裁决/补执行恰好一方成功"""

    def test_transition_moves_status(self, store):
        action = store.propose(tool_name="restart_instance")
        moved = store.transition(action.action_id, ActionStatus.PENDING, ActionStatus.APPROVED)

        assert moved is not None
        assert moved.status is ActionStatus.APPROVED
        assert moved.decided_at is not None
        assert store.get(action.action_id).status is ActionStatus.APPROVED

    def test_transition_from_wrong_status_fails(self, store):
        """from 状态不符（已被他人抢先推进）返回 None，不产生部分写"""
        action = store.propose(tool_name="restart_instance")

        # pending → approved 不存在，认领失败
        assert (
            store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED) is None
        )
        assert store.get(action.action_id).status is ActionStatus.PENDING

    def test_approved_to_executed_exactly_once(self, store):
        """approve 后的补执行认领：第二次 approved→executed 必须失败（防重放）"""
        action = store.propose(tool_name="restart_instance")
        store.transition(action.action_id, ActionStatus.PENDING, ActionStatus.APPROVED)

        first = store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED)
        assert first is not None and first.status is ActionStatus.EXECUTED

        replay = store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED)
        assert replay is None  # 状态已是 executed，条件 UPDATE 零行命中

    def test_executed_is_terminal_against_all_sources(self, store):
        action = store.propose(tool_name="t")
        store.transition(action.action_id, ActionStatus.PENDING, ActionStatus.APPROVED)
        store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED)

        # executed 终态：任何来源的状态迁移都不再命中
        assert (
            store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.REJECTED) is None
        )
        assert (
            store.transition(action.action_id, ActionStatus.PENDING, ActionStatus.APPROVED) is None
        )
        assert store.get(action.action_id).status is ActionStatus.EXECUTED


class TestTtlEnforcedOnDecision:
    """TTL 必须在裁决路径强制执行，而非只在读取时惰性判定：

    此前 decide→transition 的 UPDATE 不带 created_at 条件——只要没人调过
    list_pending 触发清理，过期动作照样能被批准并补执行（审批窗口可无限延长）。
    """

    def test_expired_action_cannot_be_approved(self, store):
        action = store.propose(tool_name="restart_instance")
        _backdate(store, action.action_id, seconds=store.ttl_seconds + 60)

        decided = store.decide(action.action_id, ActionStatus.APPROVED)
        assert decided.status is not ActionStatus.APPROVED
        # 惰性判定把过期动作标为 expired，绝不进入可执行状态
        assert decided.status is ActionStatus.EXPIRED

    def test_expired_action_cannot_transition_directly(self, store):
        action = store.propose(tool_name="restart_instance")
        _backdate(store, action.action_id, seconds=store.ttl_seconds + 60)

        assert (
            store.transition(action.action_id, ActionStatus.PENDING, ActionStatus.APPROVED) is None
        )
        assert store.get(action.action_id).status is ActionStatus.EXPIRED

    def test_fresh_action_still_decidable(self, store):
        """正向对照：窗口内裁决不受 TTL 收紧影响"""
        action = store.propose(tool_name="scale_out")
        assert store.decide(action.action_id, ActionStatus.APPROVED).status is ActionStatus.APPROVED

    def test_execute_claim_not_ttl_gated(self, store):
        """approved→executed 的补执行认领不适用 TTL 条件（裁决发生在窗口内即可）"""
        action = store.propose(tool_name="restart_instance")
        store.decide(action.action_id, ActionStatus.APPROVED)
        _backdate(store, action.action_id, seconds=store.ttl_seconds + 60)

        claimed = store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED)
        assert claimed is not None and claimed.status is ActionStatus.EXECUTED
