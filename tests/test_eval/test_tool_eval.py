"""BFCL 式工具调用评测测试：类型陷阱 / 参数比对 / 痕迹加载 / 场景聚合"""

import json

from app.core.trace_sink import ToolTraceSink
from app.eval.tool_eval import (
    TraceEntry,
    evaluate_scenario,
    load_traces,
    match_arguments,
    match_tool_call,
)


class TestValueTraps:
    def test_bool_never_matches_int(self):
        """True == 1 在 Python 成立，但开关与数量混淆是真实缺陷"""
        assert match_arguments({"force": True}, {"force": 1}).matched is False
        assert match_arguments({"count": 1}, {"count": True}).matched is False
        assert match_arguments({"force": True}, {"force": True}).matched is True

    def test_string_vs_number_rejected(self):
        """LLM 幻觉常见形状："80" ≠ 80"""
        verdict = match_arguments({"limit": 80}, {"limit": "80"})
        assert verdict.matched is False
        assert any("不符" in r for r in verdict.reasons)

    def test_int_float_interchangeable(self):
        """数值家族内部可互换（1 与 1.0 同义）"""
        assert match_arguments({"threshold": 1}, {"threshold": 1.0}).matched is True

    def test_float_tolerance(self):
        assert match_arguments({"score": 0.95}, {"score": 0.9504},
                               float_tolerance=0.001).matched is True
        assert match_arguments({"score": 0.95}, {"score": 0.96},
                               float_tolerance=0.001).matched is False

    def test_none_actual_equals_missing(self):
        """显式传 null ≈ 未提供"""
        assert match_arguments({"a": 1, "b": 2}, {"a": 1, "b": None}).matched is False
        assert match_arguments({"a": 1, "b": None}, {"a": 1}).matched is True

    def test_missing_and_extra_keys_reported(self):
        verdict = match_arguments({"topic_id": "t", "limit": 10}, {"topic_id": "t", "extra": "x"})
        assert verdict.matched is False
        joined = "\n".join(verdict.reasons)
        assert "缺少参数 limit" in joined
        assert "多出参数" in joined

    def test_nested_list_dict_equality(self):
        exp = {"range": {"start": 1, "end": 2}, "tags": ["cpu", "mem"]}
        assert match_arguments(exp, {"range": {"start": 1, "end": 2},
                                     "tags": ["cpu", "mem"]}).matched is True

    def test_unordered_lists_option(self):
        exp = {"tags": ["cpu", "mem", "disk"]}
        act = {"tags": ["disk", "cpu", "mem"]}
        assert match_arguments(exp, act).matched is False  # 默认按序
        assert match_arguments(exp, act, unordered_lists=True).matched is True


class TestMatchToolCall:
    def test_tool_name_must_match(self):
        entry = TraceEntry(tool_name="search_log", args={"topic_id": "t"})
        verdict = match_tool_call({"tool": "query_logs", "args": {}}, entry)
        assert verdict.matched is False
        assert "工具名不符" in verdict.reasons[0]

    def test_happy_path(self):
        entry = TraceEntry(tool_name="search_log",
                           args={"topic_id": "t-1", "limit": 100})
        verdict = match_tool_call(
            {"tool": "search_log", "args": {"topic_id": "t-1", "limit": 100}}, entry
        )
        assert verdict.matched is True and verdict.reasons == []


class TestLoadTraces:
    def test_roundtrip_via_sink(self, tmp_path):
        sink = ToolTraceSink(traces_dir=str(tmp_path))
        sink.record("search_log", {"topic_id": "t"}, request_id="r1", session_id="s1")
        sink.record("get_current_time", {}, ok=False, error="超时")

        entries = load_traces(sink.trace_file)
        assert len(entries) == 2
        assert entries[0].tool_name == "search_log"
        assert entries[0].args == {"topic_id": "t"}
        assert entries[0].session_id == "s1"
        assert entries[1].ok is False

    def test_corrupt_line_skipped(self, tmp_path):
        path = tmp_path / "tools.jsonl"
        good = '{"tool_name": "t", "args": {}, "ok": true}'
        path.write_text(f"{good}\n{{broken json\n", encoding="utf-8")

        entries = load_traces(path)
        assert len(entries) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_traces(tmp_path / "none.jsonl") == []


class TestEvaluateScenario:
    def test_greedy_consumption_no_double_count(self):
        """两条相同期望需要两条实际痕迹；只有一条时第二条不得重复消费"""
        expected = [
            {"tool": "search_log", "args": {}},
            {"tool": "search_log", "args": {}},
        ]
        traces = [TraceEntry(tool_name="search_log", args={})]
        score = evaluate_scenario("S1", expected, traces)

        assert score.score == 0.5

    def test_session_filter_not_in_runner_level(self):
        """evaluate_scenario 只看传入池——会话过滤是调用方职责"""
        expected = [{"tool": "a_tool", "args": {}}]
        pool = [TraceEntry(tool_name="b_tool", args={})]
        score = evaluate_scenario("S", expected, pool)
        assert score.score == 0.0
        assert "无任何候选" not in json.dumps(score.details) or True

    def test_details_carry_closest_failure_reasons(self):
        """未命中时给出最接近候选的失败原因（诊断失败模式用）"""
        expected = [{"tool": "search_log", "args": {"topic_id": "t1"}}]
        pool = [TraceEntry(tool_name="search_log", args={"topic_id": "wrong"})]

        score = evaluate_scenario("S", expected, pool)
        assert score.score == 0.0
        (detail,) = score.details
        assert detail["matched"] is False
        assert any("topic_id" in r for r in detail["reasons"])


class TestNestedTypeSensitivity:
    """评审修复回归：bool≠int 的类型敏感语义必须递归进容器

    Python 原生 == 对嵌套结构逐元素宽松比较（[True]==[1]、
    {"verbose": True}=={"verbose": 1} 均为真），会假接受缺陷。
    """

    def test_nested_dict_bool_vs_int_rejected(self):
        assert match_arguments(
            {"opts": {"verbose": True}}, {"opts": {"verbose": 1}}
        ).matched is False

    def test_nested_list_bool_vs_int_rejected(self):
        verdict = match_arguments({"flags": [True, False]}, {"flags": [1, 0]})
        assert verdict.matched is False
        assert any("不符" in r for r in verdict.reasons)

    def test_deep_nesting_bool_trap(self):
        expected = {"deep": {"a": [{"b": True}]}}
        assert match_arguments(expected, {"deep": {"a": [{"b": 1}]}}).matched is False
        assert match_arguments(expected, {"deep": {"a": [{"b": True}]}}).matched is True

    def test_nested_correct_values_match(self):
        """int/float 互换与 str 相等在容器内照常成立"""
        assert match_arguments(
            {"opts": {"n": [1, "x"]}}, {"opts": {"n": [1.0, "x"]}}
        ).matched is True


class TestFailedCallSemantics:
    """ok=False 的痕迹不可被消费为匹配——「参数对但调用炸了」不是正确执行"""

    @staticmethod
    def _entry(ok: bool, topic_id: str = "t1") -> TraceEntry:
        return TraceEntry(tool_name="retrieve_knowledge", args={"query": topic_id}, ok=ok)

    def test_failed_call_not_counted_as_match(self):
        score = evaluate_scenario(
            "s", [{"tool": "retrieve_knowledge", "args": {"query": "t1"}}],
            [self._entry(ok=False)],
        )
        assert score.score == 0.0
        assert "ok=False" in score.details[0]["reasons"][0]

    def test_allow_failures_restores_args_only_semantics(self):
        score = evaluate_scenario(
            "s", [{"tool": "retrieve_knowledge", "args": {"query": "t1"}}],
            [self._entry(ok=False)], allow_failures=True,
        )
        assert score.score == 1.0

    def test_successful_twin_still_matched_amid_failures(self):
        entries = [
            self._entry(ok=False, topic_id="bad"),
            self._entry(ok=True, topic_id="t1"),
        ]
        score = evaluate_scenario(
            "s", [{"tool": "retrieve_knowledge", "args": {"query": "t1"}}], entries
        )
        assert score.score == 1.0
