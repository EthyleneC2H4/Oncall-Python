"""C2 告警 webhook 接入通道测试：解析归一化 / 去重 / 后台诊断排队 / resolved 学习"""

import pytest

from app.api import alerts
from app.config import config

ALERTMANAGER_PAYLOAD = {
    "groupKey": "grp-1",
    "status": "firing",
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "HighCPUUsage",
                "severity": "critical",
                "instance": "node-1",
                "job": "api-server",
            },
            "annotations": {"description": "CPU 使用率超过 90%"},
            "fingerprint": "fp-1",
        }
    ],
}


@pytest.fixture
def webhook_env(monkeypatch):
    """清空去重表、关掉 token、拦截后台诊断（防真实 LLM 调用）；返回诊断调用记录"""
    alerts._dedup.clear()
    monkeypatch.setattr(config, "alert_webhook_token", "")
    calls: list[tuple[str, str]] = []

    async def fake_diagnosis(task_text: str, session_id: str) -> None:
        calls.append((task_text, session_id))

    monkeypatch.setattr(alerts, "run_webhook_diagnosis", fake_diagnosis)
    yield calls
    alerts._dedup.clear()


class TestExtractAndNormalize:
    def test_alerts_array_preferred(self):
        payload = {"alerts": [{"labels": {"alertname": "A"}}, "junk", {"no_name": 1}]}
        # 提取层只做结构过滤（保留全部 dict）；无 alertname 的条目由归一化层丢弃
        assert alerts._extract_alerts(payload) == [{"labels": {"alertname": "A"}}, {"no_name": 1}]

    def test_single_alert_object_fallback(self):
        assert alerts._extract_alerts({"alertname": "A"}) == [{"alertname": "A"}]
        assert (
            alerts._extract_alerts({"labels": {"alertname": "A"}})[0]["labels"]["alertname"] == "A"
        )

    def test_unrecognized_payload_yields_empty(self):
        assert alerts._extract_alerts({"foo": "bar"}) == []
        assert alerts._extract_alerts({"alerts": ["not-a-dict"]}) == []

    def test_normalize_missing_alertname_returns_none(self):
        raw = {"labels": {"severity": "critical"}, "annotations": {}}
        assert alerts._normalize(raw, "firing") is None

    def test_normalize_aliases_and_group_status(self):
        raw = {
            "labels": {"alertname": "SlowResponse", "job": "gateway"},
            "annotations": {"summary": "响应变慢"},
        }
        alert = alerts._normalize(raw, "firing")
        assert alert == {
            "alertname": "SlowResponse",
            "status": "firing",  # 条目无 status 时回落到组状态
            "severity": "warning",  # 缺省严重级
            "instance": "",
            "service": "gateway",  # job 别名
            "description": "响应变慢",  # summary 别名
            "root_cause": "",
            "resolution": "",
            "starts_at": "",
            "fingerprint": "",
        }

    def test_normalize_entry_status_overrides_group(self):
        raw = {"labels": {"alertname": "A"}, "status": "RESOLVED"}
        assert alerts._normalize(raw, "firing")["status"] == "resolved"


class TestBuildDiagnosisTask:
    def test_numbered_lines_with_context(self):
        task = alerts.build_diagnosis_task(
            [
                {
                    "alertname": "HighCPUUsage",
                    "severity": "critical",
                    "instance": "node-1",
                    "service": "api",
                    "description": "CPU 过高",
                    "starts_at": "2026-08-25T00:00:00Z",
                    "status": "firing",
                    "root_cause": "",
                    "resolution": "",
                    "fingerprint": "fp",
                }
            ]
        )
        assert "1. [critical] HighCPUUsage @ node-1" in task
        assert "服务: api" in task
        assert "描述: CPU 过高" in task
        assert "根因分析" in task  # 任务目标说明


class TestWebhookEndpoint:
    def test_missing_token_rejected_with_401(self, test_app, webhook_env, monkeypatch):
        monkeypatch.setattr(config, "alert_webhook_token", "s3cret")
        resp = test_app.post("/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD)
        assert resp.status_code == 401

        bad = test_app.post(
            "/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD, headers={"X-Alert-Token": "wrong"}
        )
        assert bad.status_code == 401
        assert webhook_env == []  # 未触发任何诊断

    def test_empty_token_disables_auth(self, test_app, webhook_env, monkeypatch):
        monkeypatch.setattr(config, "alert_webhook_token", "")
        resp = test_app.post("/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD)
        assert resp.json()["data"]["accepted"] == 1

    @pytest.mark.parametrize(
        "body",
        [
            '{"broken',  # 非法 JSON
            '"just a string"',  # 合法 JSON 但不是对象
            "[1, 2]",  # 顶层数组同样拒绝
        ],
        ids=["bad-json", "json-string", "json-array"],
    )
    def test_malformed_payload_rejected_400(self, test_app, webhook_env, body):
        resp = test_app.post(
            "/api/alerts/webhook",
            content=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert webhook_env == []

    def test_no_extractable_alerts_rejected_422(self, test_app, webhook_env):
        resp = test_app.post("/api/alerts/webhook", json={"foo": "bar"})
        assert resp.status_code == 422

    def test_accepted_batch_queues_one_background_diagnosis(self, test_app, webhook_env):
        resp = test_app.post("/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["code"] == 202  # 业务码：已受理（HTTP 层沿用仓库 200+code 约定）
        data = resp.json()["data"]
        assert data["accepted"] == 1
        assert data["suppressed"] == 0
        assert data["session_id"].startswith("webhook-")

        ((task_text, session_id),) = webhook_env  # BackgroundTasks 在 post 返回前已执行
        assert "HighCPUUsage" in task_text
        assert session_id == data["session_id"]

    def test_duplicate_fingerprint_suppressed_within_window(self, test_app, webhook_env):
        first = test_app.post("/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD).json()["data"]
        second = test_app.post("/api/alerts/webhook", json=ALERTMANAGER_PAYLOAD).json()["data"]

        assert first["accepted"] == 1
        assert second["accepted"] == 0
        assert second["suppressed"] == 1
        assert len(webhook_env) == 1  # 只诊断一次

    def test_resolved_with_explicit_root_cause_learns_kg(self, test_app, webhook_env, monkeypatch):
        from app.services.knowledge_graph_service import knowledge_graph_service

        recorded: dict = {}

        def fake_update(incident: dict) -> dict:
            recorded.update(incident)
            return {"nodes_added": 2}

        monkeypatch.setattr(knowledge_graph_service, "update_from_incident", fake_update)
        payload = {
            "status": "resolved",
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {"alertname": "HighMemoryUsage"},
                    "annotations": {"root_cause": "内存泄漏", "resolution": "回滚版本并重启"},
                },
                {"status": "resolved", "labels": {"alertname": "CascadingTimeout"}},
            ],
        }
        data = test_app.post("/api/alerts/webhook", json=payload).json()["data"]
        assert data["accepted"] == 0
        assert data["resolved_learned"] == ["HighMemoryUsage"]
        assert recorded["alert_type"] == "HighMemoryUsage"
        assert recorded["root_cause"] == "内存泄漏"
        assert recorded["cascade_alerts"] == ["CascadingTimeout"]  # 同批其余告警作为级联

    def test_resolved_without_explicit_annotations_not_learned(
        self, test_app, webhook_env, monkeypatch
    ):
        """描述文本不是根因——不喂给图谱（防污染）"""
        from app.services.knowledge_graph_service import knowledge_graph_service

        called = False

        def fake_update(incident: dict) -> dict:
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(knowledge_graph_service, "update_from_incident", fake_update)
        payload = {
            "status": "resolved",
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {"alertname": "HighDiskUsage"},
                    "annotations": {
                        "description": "磁盘占用恢复"
                    },  # 只有描述，无 root_cause/resolution
                }
            ],
        }
        data = test_app.post("/api/alerts/webhook", json=payload).json()["data"]
        assert data["resolved_learned"] == []
        assert called is False
