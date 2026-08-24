"""CI 评测套件测试：bfcl / gaia / judge 的运行、跳过与门禁语义"""

import json
from pathlib import Path

import pytest

from app.eval.ci_runner import SUITE_DEFAULT_PATHS, CIEvalRunner
from app.eval.dataset_registry import save_dataset


@pytest.fixture
def runner(tmp_path):
    return CIEvalRunner(output_dir=str(tmp_path / "results"))


class TestBfclSuite:
    def test_skipped_when_no_traces(self, runner):
        report = runner.run_bfcl(traces_path="/nonexistent/tools.jsonl")
        assert report["skipped"] is True
        assert report["passed"] is True  # 跳过 ≠ 失败

    def test_skipped_when_dataset_unversioned(self, runner, tmp_path):
        """legacy 裸列表金标拒载 → skipped（版本化契约贯穿到 CI）"""
        traces = tmp_path / "tools.jsonl"
        entry = {"tool_name": "search_log", "args": {"topic_id": "t"}, "ok": True}
        traces.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        dataset = tmp_path / "expectations.json"
        dataset.write_text(json.dumps([{"id": "S1", "expected_tool_calls": []}]),
                           encoding="utf-8")

        report = runner.run_bfcl(traces_path=str(traces), dataset_path=str(dataset))
        assert report["skipped"] is True
        assert "版本化" in report["reason"]

    def test_happy_path_scores_scenarios(self, runner, tmp_path):
        traces = tmp_path / "tools.jsonl"
        lines = [
            {"tool_name": "search_log", "args": {"topic_id": "t1", "limit": 10},
             "ok": True, "session_id": "s1"},
            {"tool_name": "get_current_time", "args": {}, "ok": True, "session_id": "s1"},
        ]
        traces.write_text(
            "\n".join(json.dumps(entry) for entry in lines) + "\n", encoding="utf-8"
        )

        dataset = tmp_path / "expectations.json"
        save_dataset(dataset, [
            {"id": "perfect", "expected_tool_calls": [
                {"tool": "search_log", "args": {"topic_id": "t1", "limit": 10}},
                {"tool": "get_current_time", "args": {}},
            ]},
            {"id": "flawed", "expected_tool_calls": [
                {"tool": "search_log", "args": {"topic_id": "wrong-id"}},
            ]},
        ], version="v1")

        report = runner.run_bfcl(traces_path=str(traces), dataset_path=str(dataset))

        assert not report.get("skipped")
        assert report["summary"]["scenarios"] == 2
        assert report["summary"]["dataset_version"] == "v1"
        assert report["summary"]["fully_matched"] == 1
        # flawed 场景的期望调用与任何痕迹都不符（贪心匹配无部分分）
        # → perfect 1.0 + flawed 0.0 → 均值 0.5，低于门禁 0.8 判 FAILED
        assert report["summary"]["avg_tool_match"] == pytest.approx(0.5)
        assert report["passed"] is False

    def test_report_file_written(self, runner, tmp_path):
        report = runner.run_bfcl(traces_path="/nonexistent")
        saved = Path(runner.output_dir) / f"ci_bfcl_{int(report['timestamp'])}.json"
        assert saved.exists()


class TestGaiaSuite:
    def test_skipped_without_answers(self, runner):
        report = runner.run_gaia(answers_path="/nonexistent/answers.json")
        assert report["skipped"] is True

    def test_grading_with_weights_and_missing(self, runner, tmp_path):
        answers = tmp_path / "latest_answers.json"
        answers.write_text(json.dumps({
            "TC1": "根因是内存泄漏，建议导出内存快照",
            "TC3": "完全跑题的回答",
        }), encoding="utf-8")

        dataset = tmp_path / "task_cases.json"
        save_dataset(dataset, [
            {"id": "TC1", "required_evidence": ["内存泄漏"], "weight": 1.0},
            {"id": "TC2", "required_evidence": ["磁盘"], "weight": 1.0},   # 无答案→跳过
            {"id": "TC3", "required_evidence": ["清理日志", "扩容"], "weight": 1.0},
        ], version="v4")

        report = runner.run_gaia(answers_path=str(answers), dataset_path=str(dataset))

        summary = report["summary"]
        assert summary["answered_cases"] == 2
        assert summary["missing_answers"] == 1
        assert summary["dataset_version"] == "v4"
        # TC1 exact(1.0) + TC3 wrong(0.0)，均值 0.5 < 阈值 0.7
        assert report["passed"] is False

    def test_passes_above_threshold(self, runner, tmp_path):
        answers = tmp_path / "a.json"
        answers.write_text(json.dumps({"G1": "提到 内存泄漏 即可"}), encoding="utf-8")
        dataset = tmp_path / "c.json"
        save_dataset(dataset, [{"id": "G1", "required_evidence": ["内存泄漏"]}], "v1")

        report = runner.run_gaia(answers_path=str(answers), dataset_path=str(dataset))
        assert report["passed"] is True


class TestJudgeSuite:
    async def test_skipped_without_pairs(self, runner):
        report = await runner.run_judge(pairs_path="/nonexistent/pairs.json")
        assert report["skipped"] is True

    async def test_runs_pairwise_and_reports_win_rate(self, runner, tmp_path, monkeypatch):
        pairs = tmp_path / "pairs.json"
        pairs.write_text(json.dumps([
            {"id": "P1", "question": "q", "answer_a": "a", "answer_b": "b"},
            {"id": "P2", "question": "q", "answer_a": "a", "answer_b": "b"},
        ]), encoding="utf-8")

        from app.eval import llm_judge as judge_module

        verdicts = iter([{"winner": "A"}, {"winner": "tie"}])

        class FakeJudge:
            async def judge_pairwise(self, question, answer_a, answer_b):
                return next(verdicts)

        monkeypatch.setattr(judge_module, "llm_judge", FakeJudge())

        report = await runner.run_judge(pairs_path=str(pairs))
        assert report["summary"]["a_wins"] == 1
        assert report["summary"]["ties"] == 1


def test_suite_default_paths_registered():
    """三个套件都有默认路径声明（CLI 免参可跑）"""
    for suite in ("bfcl", "gaia", "judge"):
        assert suite in SUITE_DEFAULT_PATHS
