"""GAIA 式任务分级评测测试：证据命中定级 / 权重聚合"""

from app.eval.task_eval import (
    TaskGrade,
    evaluate_case,
    grade_answer,
    summarize,
)


class TestGradeAnswer:
    def test_all_required_hit_is_exact(self):
        v = grade_answer("根因是内存泄漏，建议导出内存快照分析",
                         ["内存泄漏", "内存快照"])
        assert v.grade is TaskGrade.EXACT and v.score == 1.0

    def test_partial_hit_is_partial(self):
        # 命中率 2/3 ≈ 0.67 ≥ 默认阈值 0.5 → PARTIAL（1/3 的情形见下方阈值用例）
        v = grade_answer("可能是内存泄漏，也可能磁盘写满",
                         ["内存泄漏", "磁盘写满", "网络抖动"])
        assert v.grade is TaskGrade.PARTIAL
        assert v.score == 0.5
        assert v.hit_required == ["内存泄漏", "磁盘写满"]

    def test_no_hit_is_wrong(self):
        v = grade_answer("一切正常", ["内存泄漏", "OOM"])
        assert v.grade is TaskGrade.WRONG and v.score == 0.0

    def test_forbidden_evidence_forces_wrong(self):
        """即使必需证据全中，出现禁词也判 0（如错误处置建议）"""
        v = grade_answer(
            "根因是内存泄漏，直接 rm -rf 清理即可",
            ["内存泄漏"],
            forbidden_evidence=["rm -rf"],
        )
        assert v.grade is TaskGrade.WRONG
        assert v.hit_forbidden == ["rm -rf"]

    def test_empty_required_not_punished(self):
        v = grade_answer("任意回答", [])
        assert v.grade is TaskGrade.EXACT and v.score == 1.0

    def test_case_insensitive_match(self):
        v = grade_answer("Root cause: MEMORY LEAK detected", ["memory leak"])
        assert v.grade is TaskGrade.EXACT

    def test_partial_threshold_configurable(self):
        """命中率 1/3 在默认阈值下 wrong；阈值调低后 partial"""
        required = ["a", "b", "c"]
        assert grade_answer("有 a", required).grade is TaskGrade.WRONG
        v = grade_answer("有 a", required, partial_threshold=0.3)
        assert v.grade is TaskGrade.PARTIAL


class TestEvaluateCase:
    def test_schema_fields_applied(self):
        case = {
            "id": "TC009",
            "required_evidence": ["清理日志"],
            "weight": 2.0,
        }
        v = evaluate_case(case, "建议清理日志文件")
        assert v.case_id == "TC009"
        assert v.weight == 2.0
        assert v.score == 1.0  # 原始等级分不含权重

    def test_weight_default_one(self):
        v = evaluate_case({"id": "X", "required_evidence": []}, "答案")
        assert v.weight == 1.0


class TestSummarize:
    def test_weighted_average(self):
        verdicts = [
            evaluate_case({"id": "A", "required_evidence": ["x"]}, "含 x"),   # 1.0 × w1
            evaluate_case({"id": "B", "required_evidence": [], "weight": 3.0}, ""),  # 1.0 × 3
            evaluate_case({"id": "C", "required_evidence": ["y"], "weight": 2.0}, "没有命中"),
        ]
        summary = summarize(verdicts)

        # 加权均值 = (1×1 + 1×3 + 0×2) / (1+3+2) = 4/6
        assert summary["avg_score"] == round(4 / 6, 4)
        assert summary["total"] == 3
        assert summary["grades"] == {"exact": 2, "wrong": 1}

    def test_empty_batch(self):
        assert summarize([])["total"] == 0
        assert summarize([])["avg_score"] == 0.0
