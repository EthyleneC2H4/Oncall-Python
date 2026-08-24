"""parse_plan 容错矩阵 —— 任何 LLM 输出都能退化为可执行计划

解析阶梯：passthrough → dict/steps → 围栏 JSON → 括号扫描 →
截断抢救 → literal_eval → 行模式兜底。旧 List[str] 行为永远可表示。
"""

import pytest

from app.models.plan import PlanStep, StructuredPlan, parse_plan, plan_to_legacy_strings


class LegacyPlanStub:
    """模拟旧 pydantic Plan / 各家 provider 的结构化输出模型（带 .steps 属性）"""

    def __init__(self, steps):
        self.steps = steps


CLEAN_PLAN = {
    "steps": [
        {
            "id": "s1",
            "description": "查询错误日志",
            "tool": "search_log",
            "args": {"topic_id": "t-1", "start_time": 1, "end_time": 2},
            "depends_on": [],
            "expected_evidence": "日志",
        },
        {
            "id": "s2",
            "description": "综合分析",
            "depends_on": ["s1"],
        },
    ]
}


class TestStructuredInputs:
    def test_clean_json_object_with_tool_binding(self):
        plan = parse_plan(CLEAN_PLAN)

        assert plan.source_format == "structured"
        assert [s.description for s in plan.steps] == ["查询错误日志", "综合分析"]
        first = plan.steps[0]
        assert first.id == "s1"
        assert first.tool == "search_log"
        assert first.args == {"topic_id": "t-1", "start_time": 1, "end_time": 2}
        assert plan.steps[1].depends_on == ["s1"]

    def test_fenced_json_stripped(self):
        raw = f"```json\n{_json_dumps(CLEAN_PLAN)}\n```"
        plan = parse_plan(raw)
        assert plan.source_format == "fenced_json"
        assert len(plan.steps) == 2

    def test_json_embedded_in_prose(self):
        raw = '好的，计划如下：{"steps": [{"description": "查日志"}]} 请确认'
        plan = parse_plan(raw)
        assert [s.description for s in plan.steps] == ["查日志"]

    def test_truncated_json_salvages_first_complete_step(self):
        raw = '{"steps": [{"description": "查日志"}, {"des'
        plan = parse_plan(raw)

        assert plan.source_format == "truncated_json"
        assert [s.description for s in plan.steps] == ["查日志"]

    def test_single_quote_python_literal(self):
        plan = parse_plan("['查询日志', '生成报告']")
        assert plan.legacy_strings == ["查询日志", "生成报告"]

    def test_list_of_plain_strings_passthrough(self):
        plan = parse_plan(["查询日志", "生成报告"])
        assert plan.source_format == "structured"
        assert plan.legacy_strings == ["查询日志", "生成报告"]

    def test_dict_with_steps_key(self):
        plan = parse_plan({"steps": ["a", "b"]})
        assert plan.legacy_strings == ["a", "b"]

    def test_single_step_dict_with_description(self):
        plan = parse_plan({"description": "单步任务", "tool": "retrieve_knowledge"})
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "retrieve_knowledge"

    def test_planstep_and_structuredplan_passthrough_identity(self):
        structured = parse_plan(["x"])
        assert parse_plan(structured) is structured

        step = PlanStep(description="y")
        parsed = parse_plan(step)
        assert parsed.steps == [step]

    def test_steps_bearing_objects_delegated(self):
        """旧 pydantic Plan / provider 结构化模型：按 .steps 值递归解析"""
        plan = parse_plan(LegacyPlanStub(["查询日志", "生成报告"]))
        assert plan.legacy_strings == ["查询日志", "生成报告"]


class TestDegradedInputs:
    def test_none_yields_empty_plan(self):
        assert parse_plan(None).steps == []
        assert parse_plan(None).source_format == "empty"

    @pytest.mark.parametrize("blank", ["", "   \n\t"])
    def test_blank_string_yields_empty_plan(self, blank):
        assert parse_plan(blank).steps == []

    def test_scalar_becomes_single_line_step(self):
        plan = parse_plan(42)
        assert plan.legacy_strings == ["42"]

    def test_unknown_dict_shape_preserved_as_line(self):
        plan = parse_plan({"foo": 1})
        assert len(plan.steps) == 1
        assert "foo" in plan.steps[0].description

    def test_numbered_prefixes_stripped_in_line_mode(self):
        raw = "步骤1: 查看告警\n步骤2：查询日志\n3. 重启服务\n- 检查配置\n* 巡检指标"
        plan = parse_plan(raw)

        assert plan.source_format == "lines"
        assert plan.legacy_strings == ["查看告警", "查询日志", "重启服务", "检查配置", "巡检指标"]

    def test_plain_lines_without_prefix(self):
        plan = parse_plan("查日志\n查指标")
        assert plan.legacy_strings == ["查日志", "查指标"]

    def test_never_raises_on_hostile_object(self):
        """总函数契约：任意对象都不抛异常"""
        class Hostile:
            def __str__(self):
                raise RuntimeError("拒绝被字符串化")

        # __str__ 都炸的对象也要兜住（parse_plan 外层 try 兜底）
        plan = parse_plan(Hostile())
        assert isinstance(plan, StructuredPlan)


class TestNormalization:
    def test_empty_ids_filled_positionally(self):
        plan = parse_plan({"steps": [{"description": "一"}, {"description": "二"}, {"description": "三"}]})
        assert [s.id for s in plan.steps] == ["1", "2", "3"]

    def test_duplicate_ids_renumbered_but_explicit_kept(self):
        raw = {"steps": [
            {"description": "一"},
            {"id": "dup", "description": "二"},
            {"id": "dup", "description": "三"},
        ]}
        plan = parse_plan(raw)
        assert [s.id for s in plan.steps] == ["1", "dup", "3"]

    def test_empty_id_backfill_never_collides_with_explicit_ids(self):
        """回填 id 从最小未占用整数取，不得与保留的显式 id 撞车（曾盲取位置号）"""
        raw = {"steps": [{"id": "2", "description": "显式二号"}, {"description": "缺号"}]}
        plan = parse_plan(raw)
        ids = [s.id for s in plan.steps]
        assert len(ids) == len(set(ids)) == 2
        assert "2" in ids  # 显式 id 原样保留

    def test_dangling_depends_on_pruned_after_filter(self):
        """中段步骤被滤空后，指向它的 depends_on 引用必须同步清理"""
        raw = {"steps": [
            {"id": "1", "description": "首步"},
            {"id": "2", "tool": "search_log"},  # 缺 description → 被剔除
            {"id": "3", "description": "尾步", "depends_on": ["2"]},
        ]}
        plan = parse_plan(raw)
        assert [s.description for s in plan.steps] == ["首步", "尾步"]
        tail = plan.steps[1]
        assert tail.id == "3"
        assert tail.depends_on == []  # 悬挂引用已清理，而非指向不存在的 id

    def test_numeric_id_variant_no_self_dependency(self):
        """数字 id 变体：滤空中段后重排不得把陈旧引用变成自环"""
        raw = {"steps": [
            {"id": "1", "description": "a"},
            {"id": "2"},  # 空描述被剔除
            {"description": "c", "depends_on": ["2"]},
        ]}
        steps = parse_plan(raw).steps
        assert all("2" not in s.depends_on for s in steps)

    def test_decimal_prefixed_lines_not_truncated(self):
        """行模式兜底不得把小数开头的正文截头（"2.4GHz" 曾变 "4GHz"）"""
        raw = "检查无线控制器\n2.4GHz 频段信道利用率过高\n1.5G 内存占用过高"
        plan = parse_plan(raw)
        assert plan.legacy_strings == ["检查无线控制器", "2.4GHz 频段信道利用率过高", "1.5G 内存占用过高"]

    def test_tight_compact_numbering_still_stripped(self):
        """收紧正则后紧凑中文编号（无空格）仍要剥掉"""
        plan = parse_plan("1.检查日志\n2.查指标")
        assert plan.legacy_strings == ["检查日志", "查指标"]

    def test_empty_description_steps_filtered_before_id_fill(self):
        raw = {"steps": [{"description": "   "}, {"description": "有效步骤"}]}
        plan = parse_plan(raw)
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "1"

    def test_field_coercions(self):
        """str args → {value}、str depends_on → [ref]（引用需指向现存 id 才保留）"""
        raw = {"steps": [
            {
                "description": "x",
                "args": "原始值",
                "depends_on": "s0",
                "hallucinated_field": "忽略我",
            },
            {"id": "s0", "description": "前置步骤"},
        ]}
        (step, _) = parse_plan(raw).steps
        assert step.args == {"value": "原始值"}
        assert step.depends_on == ["s0"]
        assert not hasattr(step, "hallucinated_field")


class TestLegacyBridge:
    def test_plan_to_legacy_strings_all_shapes(self):
        structured = parse_plan(CLEAN_PLAN)
        assert plan_to_legacy_strings(structured) == ["查询错误日志", "综合分析"]
        assert plan_to_legacy_strings(["a", "b"]) == ["a", "b"]
        assert plan_to_legacy_strings('{"steps": [{"description": "查"}]}') == ["查"]
        assert plan_to_legacy_strings(LegacyPlanStub(["s"])) == ["s"]


def _json_dumps(data) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)
