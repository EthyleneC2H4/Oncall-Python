"""负反馈回填金标集测试：去重 / 版本递增 / API 闭环"""

import json

import pytest

import app.api.feedback as feedback_module
from app.api.feedback import backfill_negative_case
from app.models.diagnosis_report import FeedbackRecord


def _record(session="s1", idx=0, comment="CPU 持续 95%", cause="死循环"):
    return FeedbackRecord(
        session_id=session,
        message_index=idx,
        feedback_type="negative",
        comment=comment,
        actual_root_cause=cause,
    )


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    """临时数据集目录 + chdir（backfill 默认走相对路径 eval/datasets）"""
    d = tmp_path / "eval" / "datasets"
    d.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return d


class TestBackfillUnit:
    def test_backfills_with_version_and_origin(self, datasets_dir):
        info = backfill_negative_case(_record())

        assert info["backfilled"] is True
        assert info["version"] == "v1"

        raw = json.loads((datasets_dir / "negative_cases.json").read_text(encoding="utf-8"))
        assert raw["version"] == "v1"
        (case,) = raw["cases"]
        assert case["origin"] == "feedback"
        assert case["category"] == "feedback"
        assert case["required_evidence"] == ["死循环"]

    def test_duplicate_not_backfilled_twice(self, datasets_dir):
        backfill_negative_case(_record())
        second = backfill_negative_case(
            _record(comment="CPU   持续 95%", cause=" 死循环 ")  # 空白差异不算新 case
        )

        assert second["backfilled"] is False
        assert second["reason"] == "duplicate"

        raw = json.loads((datasets_dir / "negative_cases.json").read_text(encoding="utf-8"))
        assert len(raw["cases"]) == 1

    def test_version_bumps_on_each_new_case(self, datasets_dir):
        v1 = backfill_negative_case(_record(idx=1))
        v2 = backfill_negative_case(_record(idx=2, comment="磁盘写满", cause="日志未轮转"))

        assert (v1["version"], v2["version"]) == ("v1", "v2")

    def test_legacy_bare_list_migrated_and_bumped(self, datasets_dir):
        """存量 legacy 文件首次回填时迁移为信封并从 v1 起"""
        path = datasets_dir / "negative_cases.json"
        legacy = [{"id": "OLD", "query": "旧用例", "actual_root_cause": "配置错误"}]
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

        info = backfill_negative_case(_record())
        assert info["backfilled"] is True
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["version"] == "v1"
        assert [c["id"] for c in raw["cases"]] == ["OLD", "FB_s1_0"]

    def test_missing_fields_skipped(self, datasets_dir):
        no_comment = FeedbackRecord(
            session_id="s", message_index=0,
            feedback_type="negative", comment="", actual_root_cause="根因",
        )
        info = backfill_negative_case(no_comment)
        assert info["backfilled"] is False
        assert not (datasets_dir / "negative_cases.json").exists()


class TestFeedbackAPI:
    def test_positive_feedback_never_touches_dataset(self, datasets_dir):
        """positive 无实际根因：即便误调 helper 也绝不写数据集"""
        record = FeedbackRecord(
            session_id="s", message_index=0, feedback_type="positive",
            comment="很准", actual_root_cause="",
        )
        info = backfill_negative_case(record)

        assert info["backfilled"] is False
        assert not (datasets_dir / "negative_cases.json").exists()


class TestFeedbackAPIIntegration:
    async def test_post_negative_creates_dataset_entry(self, tmp_path, monkeypatch, test_app):
        """API 闭环：POST /api/feedback negative → 金标集出现 origin=feedback 用例"""
        datasets = tmp_path / "eval" / "datasets"
        datasets.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)  # backfill 解析 cwd 相对路径，必须钉进临时目录
        # FEEDBACK_FILE 锚定仓库根（相对 __file__），chdir 管不到 → 显式重定向
        monkeypatch.setattr(
            feedback_module, "FEEDBACK_FILE", str(tmp_path / "feedback.json")
        )

        resp = test_app.post("/api/feedback", json={
            "session_id": "sess-9",
            "message_index": 3,
            "feedback_type": "negative",
            "comment": "服务响应慢",
            "actual_root_cause": "数据库连接池耗尽",
        })

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["dataset_backfill"]["backfilled"] is True
        assert body["dataset_backfill"]["version"] == "v1"

        raw = json.loads((datasets / "negative_cases.json").read_text(encoding="utf-8"))
        (case,) = raw["cases"]
        assert case["origin"] == "feedback"
        assert case["query"] == "服务响应慢"

    def test_post_duplicate_feedback_deduped(self, tmp_path, monkeypatch, test_app):
        datasets = tmp_path / "eval" / "datasets"
        datasets.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)  # 同上：隔离 cwd，防写穿到仓库真实数据集
        monkeypatch.setattr(
            feedback_module, "FEEDBACK_FILE", str(tmp_path / "feedback.json")
        )
        payload = {
            "session_id": "sess-1",
            "message_index": 0,
            "feedback_type": "negative",
            "comment": "内存高",
            "actual_root_cause": "泄漏",
        }
        test_app.post("/api/feedback", json=payload)
        second = test_app.post("/api/feedback", json=payload)

        assert second.status_code == 200
        data = second.json()["data"]
        assert data["dataset_backfill"]["backfilled"] is False
        assert data["dataset_backfill"]["reason"] == "duplicate"

        raw = json.loads((datasets / "negative_cases.json").read_text(encoding="utf-8"))
        assert len(raw["cases"]) == 1


class TestSanitization:
    """评审修复回归：金标集随仓库分发，外部输入必须净化后落盘"""

    def test_pii_masked_in_golden_set(self, datasets_dir):
        rec = FeedbackRecord(
            session_id="s", message_index=0, feedback_type="negative",
            comment="联系运维 13812345678 处理 CPU 持续 95%",
            actual_root_cause="死循环",
        )
        info = backfill_negative_case(rec)

        assert info["backfilled"] is True
        raw = json.loads((datasets_dir / "negative_cases.json").read_text(encoding="utf-8"))
        query = raw["cases"][0]["query"]
        assert "13812345678" not in query
        assert "138****5678" in query  # input_guard.mask_pii 生效

    def test_control_chars_and_length_bounded(self, datasets_dir):
        rec = FeedbackRecord(
            session_id="s", message_index=0, feedback_type="negative",
            comment="查询\x00带控制字符" + "长" * 600,
            actual_root_cause="根因",
        )
        backfill_negative_case(rec)
        raw = json.loads((datasets_dir / "negative_cases.json").read_text(encoding="utf-8"))
        case = raw["cases"][0]
        assert "\x00" not in case["query"]
        assert len(case["query"]) <= 500

    def test_model_rejects_oversized_fields(self):
        """pydantic 层长度上限：超长请求在 API 边界即 422"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FeedbackRecord(session_id="s" * 200, message_index=0, feedback_type="negative")
        with pytest.raises(ValidationError):
            FeedbackRecord(
                session_id="s", message_index=0, feedback_type="negative",
                actual_root_cause="x" * 501,
            )
