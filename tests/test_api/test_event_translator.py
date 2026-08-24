"""SSE 事件翻译器 golden 快照测试

钉死 /api/aiops 的旧版 SSE dict 契约：AgentEvent → legacy dict 的映射
必须与重构前 aiops_service 事件格式化函数的输出逐字段一致。
修改映射规则前，先确认前端对相应字段的依赖（只增不改原则）。

注：step_complete 的 result_preview 为 P1.1 新增键（端点 docstring 早有记载，
实际旧实现漏发）；其余字段均为历史原样。
"""

from app.agent.runtime.events import AgentEventEmitter, EventType
from app.api.event_translator import agent_event_to_legacy


def emit(event_type: EventType, **payload):
    """构造 seq=1 的测试事件"""
    return AgentEventEmitter(session_id="s1").emit(event_type, **payload)


class TestGoldenPlan:
    def test_plan_created_golden(self):
        event = emit(
            EventType.PLAN_CREATED,
            message="执行计划已制定，共 3 个步骤",
            plan=["步骤1", "步骤2", "步骤3"],
            kg_context="CPU告警: 内存泄漏",
            query_intent="DIAGNOSTIC",
        )

        assert agent_event_to_legacy(event) == {
            "type": "plan",
            "stage": "plan_created",
            "message": "执行计划已制定，共 3 个步骤",
            "plan": ["步骤1", "步骤2", "步骤3"],
            "kg_context": "CPU告警: 内存泄漏",
            "query_intent": "DIAGNOSTIC",
        }

    def test_plan_created_without_optional_context(self):
        event = emit(
            EventType.PLAN_CREATED, message="执行计划已制定，共 1 个步骤", plan=["仅一步"]
        )

        assert agent_event_to_legacy(event) == {
            "type": "plan",
            "stage": "plan_created",
            "message": "执行计划已制定，共 1 个步骤",
            "plan": ["仅一步"],
        }


class TestGoldenStepComplete:
    def test_step_complete_golden(self):
        event = emit(
            EventType.STEP_END,
            current_step="查询系统日志",
            result_preview="发现 OOM 异常堆栈",
            steps_done=2,
            remaining_steps=4,
        )

        assert agent_event_to_legacy(event) == {
            "type": "step_complete",
            "stage": "step_executed",
            "message": "步骤执行完成 (2/6)",
            "current_step": "查询系统日志",
            "result_preview": "发现 OOM 异常堆栈",
            "remaining_steps": 4,
        }


class TestGoldenStatus:
    def test_replanner_continue_golden(self):
        event = emit(EventType.REPLAN, message="评估完成，继续执行剩余步骤", remaining_steps=2)

        assert agent_event_to_legacy(event) == {
            "type": "status",
            "stage": "replanner",
            "message": "评估完成，继续执行剩余步骤",
            "remaining_steps": 2,
        }

    def test_replanner_prepare_response_golden(self):
        event = emit(EventType.REPLAN, message="评估完成，准备生成最终响应", remaining_steps=0)

        assert agent_event_to_legacy(event)["message"] == "评估完成，准备生成最终响应"

    def test_step_start_status_golden(self):
        event = emit(EventType.STEP_START, stage="executor", message="开始执行步骤")

        assert agent_event_to_legacy(event) == {
            "type": "status",
            "stage": "executor",
            "message": "开始执行步骤",
        }


class TestGoldenReport:
    def test_report_golden(self):
        event = emit(EventType.REPORT, report="# 故障诊断报告\n...")

        assert agent_event_to_legacy(event) == {
            "type": "report",
            "stage": "final_report",
            "message": "最终报告已生成",
            "report": "# 故障诊断报告\n...",
        }


class TestGoldenComplete:
    def test_execute_mode_golden(self):
        event = emit(
            EventType.COMPLETE, message="任务执行完成", response="# 报告", timed_out=False
        )

        assert agent_event_to_legacy(event) == {
            "type": "complete",
            "stage": "complete",
            "message": "任务执行完成",
            "response": "# 报告",
            "timed_out": False,
        }

    def test_diagnosis_mode_golden(self):
        """/api/aiops 契约：complete 包装为 diagnosis 形状"""
        event = emit(
            EventType.COMPLETE, message="任务执行完成", response="# 诊断报告全文", timed_out=False
        )

        assert agent_event_to_legacy(event, diagnosis_mode=True) == {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": "诊断流程完成",
            "diagnosis": {"status": "completed", "report": "# 诊断报告全文"},
        }


class TestGoldenError:
    def test_error_golden(self):
        event = emit(EventType.ERROR, message="任务执行出错: MCP 不可用")

        assert agent_event_to_legacy(event) == {
            "type": "error",
            "stage": "error",
            "message": "任务执行出错: MCP 不可用",
        }


class TestAdditiveTypes:
    def test_new_event_types_return_none(self):
        """TOKEN / TOOL_* 等新类型不属于 /api/aiops 旧契约 → 跳过"""
        for new_type in (EventType.TOKEN, EventType.TOOL_START, EventType.TOOL_END):
            event = emit(new_type, text="x")
            assert agent_event_to_legacy(event) is None

    def test_terminal_types_break_contract_keys_present(self):
        """complete/error 翻译结果必须含 type 键（SSE 流据此断开）"""
        complete = agent_event_to_legacy(emit(EventType.COMPLETE, response="", timed_out=False))
        error = agent_event_to_legacy(emit(EventType.ERROR, message="e"))

        assert complete["type"] in ("complete",)
        assert error["type"] in ("error",)


class TestPlanStructuredAdditive:
    def test_plan_structured_passthrough_adds_field(self):
        """P3 只增不改：plan_structured 作为新增可选字段透传，旧字段原样"""
        event = emit(
            EventType.PLAN_CREATED,
            message="执行计划已制定，共 1 个步骤",
            plan=["查询错误日志"],
            plan_structured=[
                {
                    "id": "1",
                    "description": "查询错误日志",
                    "tool": "search_log",
                    "args": {"topic_id": "t-1"},
                    "depends_on": [],
                    "expected_evidence": "日志",
                }
            ],
        )

        legacy = agent_event_to_legacy(event)
        # 旧契约字段原样保留
        assert legacy["type"] == "plan"
        assert legacy["stage"] == "plan_created"
        assert legacy["plan"] == ["查询错误日志"]
        # 新增字段整体透传，不做形状改写
        assert legacy["plan_structured"][0]["tool"] == "search_log"
        assert legacy["plan_structured"][0]["args"] == {"topic_id": "t-1"}

    def test_plan_without_structured_stays_golden(self):
        """未携带 plan_structured 时输出与既有 golden 完全一致（无空键污染）"""
        event = emit(EventType.PLAN_CREATED, message="执行计划已制定，共 1 个步骤", plan=["仅一步"])

        legacy = agent_event_to_legacy(event)
        assert "plan_structured" not in legacy
        assert set(legacy) == {"type", "stage", "message", "plan"}
