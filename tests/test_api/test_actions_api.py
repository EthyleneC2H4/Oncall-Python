"""高风险动作审批 API 测试：GET pending / POST approve / POST reject"""

import pytest

from app.config import config
from app.services.pending_actions import (
    ActionStatus,
    get_pending_action_store,
    reset_pending_action_store,
)


@pytest.fixture
def actions_env(tmp_path, monkeypatch):
    """待审库指向临时路径并重置单例（API 层经 get_pending_action_store 惰性重建）"""
    monkeypatch.setattr(config, "pending_actions_db_path", str(tmp_path / "api-actions.db"))
    reset_pending_action_store()
    yield get_pending_action_store()
    reset_pending_action_store()


class TestListPending:
    def test_empty_list(self, test_app, actions_env):
        resp = test_app.get("/api/actions/pending")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 0
        assert data["actions"] == []

    def test_seeded_action_listed(self, test_app, actions_env):
        action = actions_env.propose(
            tool_name="restart_instance", args={"instance_id": "i-1"}, reason="高风险"
        )

        data = test_app.get("/api/actions/pending").json()["data"]
        assert data["total"] == 1
        (row,) = data["actions"]
        assert row["action_id"] == action.action_id
        assert row["tool_name"] == "restart_instance"
        assert row["status"] == "pending"
        assert row["args"] == {"instance_id": "i-1"}


class TestRejectEndpoint:
    def test_reject_marks_terminal_state(self, test_app, actions_env):
        action = actions_env.propose(tool_name="scale_out")

        resp = test_app.post(f"/api/actions/{action.action_id}/reject")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["executed"] is False
        assert body["action"]["status"] == "rejected"

        # 终态后从待审列表消失
        assert test_app.get("/api/actions/pending").json()["data"]["total"] == 0

    def test_reject_unknown_returns_404(self, test_app, actions_env):
        resp = test_app.post("/api/actions/ghost/reject")
        assert resp.status_code == 404


class TestApproveEndpoint:
    def test_approve_unknown_returns_404(self, test_app, actions_env):
        resp = test_app.post("/api/actions/ghost/approve")
        assert resp.status_code == 404

    def test_approve_executes_local_readonly_tool(self, test_app, actions_env):
        """approve 闭环：登记 → 批准 → 真实补执行本地工具 → 结果回填"""
        action = actions_env.propose(tool_name="get_current_time", args={})

        resp = test_app.post(f"/api/actions/{action.action_id}/approve")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["executed"] is True
        assert body["error"] == ""
        assert body["result_preview"]
        # 补执行即原子认领为 executed 终态（防重复 approve 重放）
        assert body["action"]["status"] == "executed"

        refreshed = actions_env.get(action.action_id)
        assert refreshed.result_preview

    def test_approve_missing_tool_reports_gracefully(self, test_app, actions_env):
        """登记的工具在补执行时找不到：不炸接口，返回 executed=False + error"""
        action = actions_env.propose(tool_name="not_a_real_tool", args={})

        body = test_app.post(f"/api/actions/{action.action_id}/approve").json()["data"]
        assert body["executed"] is False
        assert "找不到工具" in body["error"]

    def test_approve_after_reject_does_not_execute(self, test_app, actions_env):
        action = actions_env.propose(tool_name="get_current_time", args={})
        actions_env.decide(action.action_id, ActionStatus.REJECTED)

        body = test_app.post(f"/api/actions/{action.action_id}/approve").json()["data"]
        assert body["executed"] is False
        assert body["action"]["status"] == "rejected"

    def test_double_approve_executes_exactly_once(self, test_app, actions_env):
        """重复 approve：第二次不再执行（executed 终态），返回现状态防重放"""
        action = actions_env.propose(tool_name="get_current_time", args={})

        first = test_app.post(f"/api/actions/{action.action_id}/approve").json()["data"]
        assert first["executed"] is True

        second = test_app.post(f"/api/actions/{action.action_id}/approve")
        assert second.status_code == 200
        body = second.json()["data"]
        assert body["executed"] is False
        assert body["action"]["status"] == "executed"
