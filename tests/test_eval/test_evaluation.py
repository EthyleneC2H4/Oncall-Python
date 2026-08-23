"""评测模块集成测试

测试覆盖：数据集加载、组件评测、结果汇总、LLM Judge。
"""

import json
from pathlib import Path


class TestDatasetLoading:
    """数据集加载测试"""

    def test_load_diagnostic_cases(self, sample_dataset_dir):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir=str(sample_dataset_dir))
        cases = evaluator.diagnostic_cases
        assert len(cases) == 1
        assert cases[0]["id"] == "TEST001"

    def test_load_negative_cases(self, sample_dataset_dir):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir=str(sample_dataset_dir))
        cases = evaluator.negative_cases
        assert len(cases) == 1
        assert cases[0]["category"] == "chitchat"

    def test_all_cases_combined(self, sample_dataset_dir):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir=str(sample_dataset_dir))
        assert len(evaluator.all_cases) == 2

    def test_missing_dataset_file_graceful(self, tmp_path):
        from app.eval.ragas_evaluator import RAGASEvaluator

        empty_dir = tmp_path / "empty_datasets"
        empty_dir.mkdir()
        evaluator = RAGASEvaluator(datasets_dir=str(empty_dir))
        assert evaluator.diagnostic_cases == []
        assert evaluator.negative_cases == []

    def test_all_cases_have_required_fields(self, sample_dataset_dir):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir=str(sample_dataset_dir))
        required = {"id", "category", "query", "expected_intent", "tags", "reference"}
        for case in evaluator.all_cases:
            missing = required - set(case.keys())
            assert not missing, f"{case['id']} missing fields: {missing}"


class TestComponentSummary:
    """组件评测汇总测试"""

    def test_compute_summary_empty(self):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        result = evaluator._compute_component_summary([])
        assert result == {}

    def test_compute_summary_perfect(self):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        perfect_results = [
            {
                "id": "TC001",
                "category": "easy",
                "routing": {"correct": True},
                "retrieval": {"returned": True},
                "kg_analysis": {"found": True, "root_cause_hit": True},
                "context_recall": 1.0,
                "context_precision": 1.0,
            }
        ]
        summary = evaluator._compute_component_summary(perfect_results)
        assert summary["routing_accuracy"] == 1.0
        assert summary["retrieval_rate"] == 1.0
        assert summary["kg_coverage"] == 1.0
        assert summary["kg_root_cause_accuracy"] == 1.0

    def test_compute_summary_partial(self):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        mixed = [
            {
                "id": "TC001",
                "category": "easy",
                "routing": {"correct": True},
                "retrieval": {"returned": True},
                "kg_analysis": {"found": True, "root_cause_hit": True},
                "context_recall": 1.0,
                "context_precision": 0.8,
            },
            {
                "id": "TC002",
                "category": "hard",
                "routing": {"correct": False},
                "retrieval": {"returned": False},
                "kg_analysis": {"found": False},
                "context_recall": 0.0,
                "context_precision": 0.0,
            },
        ]
        summary = evaluator._compute_component_summary(mixed)
        assert summary["routing_accuracy"] == 0.5
        assert summary["retrieval_rate"] == 0.5
        assert summary["kg_coverage"] == 0.5
        assert "by_category" in summary
        assert "easy" in summary["by_category"]
        assert "hard" in summary["by_category"]


class TestLLMJudge:
    """LLM-as-Judge 测试"""

    def test_parse_valid_json(self):
        from app.eval.llm_judge import LLMJudge

        judge = LLMJudge()
        result = judge._parse_judge_result('{"score": 5, "reason": "完全忠实"}')
        assert result["score"] == 5
        assert "reason" in result

    def test_parse_json_embedded_in_text(self):
        from app.eval.llm_judge import LLMJudge

        judge = LLMJudge()
        result = judge._parse_judge_result(
            '根据分析，评为高分。\n```json\n{"score": 4, "reason": "基本准确"}\n```'
        )
        assert result["score"] == 4

    def test_parse_fallback_numeric(self):
        from app.eval.llm_judge import LLMJudge

        judge = LLMJudge()
        result = judge._parse_judge_result("评分：3分，因为有些论据不足")
        assert result["score"] == 3

    def test_parse_invalid(self):
        from app.eval.llm_judge import LLMJudge

        judge = LLMJudge()
        result = judge._parse_judge_result("无法判断")
        assert result["score"] == 0

    def test_prompts_are_well_formed(self):
        from app.eval.llm_judge import FAITHFULNESS_PROMPT, RELEVANCY_PROMPT

        assert "{context}" in FAITHFULNESS_PROMPT
        assert "{answer}" in FAITHFULNESS_PROMPT
        assert "{question}" in RELEVANCY_PROMPT
        assert "{answer}" in RELEVANCY_PROMPT


class TestEvaluationResultPersistence:
    """评测结果持久化测试"""

    def test_save_results_creates_file(self, tmp_path):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        results = {"mode": "component", "summary": {"routing_accuracy": 0.9}}

        # 使用 monkeypatch 或直接指定输出目录
        import app.eval.ragas_evaluator as re

        original = re.Path
        re.Path = lambda x: tmp_path / x
        try:
            path = evaluator.save_results(results, filename="test_output.json")
            assert Path(path).exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["mode"] == "component"
        finally:
            re.Path = original


class TestRagasMetricBuilding:
    """RAGAS 指标构建测试"""

    def test_all_metrics_built_by_default(self):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        metrics = evaluator._build_metrics(None)
        assert len(metrics) == 4

    def test_specific_metrics(self):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        metrics = evaluator._build_metrics(["faithfulness", "context_recall"])
        assert len(metrics) == 2

    def test_unknown_metric_skipped_with_warning(self, caplog):
        from app.eval.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(datasets_dir="nonexistent")
        metrics = evaluator._build_metrics(["faithfulness", "unknown_metric"])
        # 未知指标应跳过，至少保留已知的
        assert len(metrics) >= 1
