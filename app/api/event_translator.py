"""SSE 事件翻译器 — AgentEvent → 各端点的旧版 SSE dict 契约

P1.1 将三个服务切换为统一 AgentEvent 流后，/api/aiops 的旧事件 dict 格式
由本模块负责翻译。映射规则与重构前的 aiops_service 事件格式化函数逐字段
一致，由 tests/test_api/test_event_translator.py 的 golden 快照钉死：

- PLAN_CREATED → {"type":"plan","stage":"plan_created",...}
- STEP_END     → {"type":"step_complete","stage":"step_executed",...}
- STEP_START   → {"type":"status",...}
- REPLAN       → {"type":"status","stage":"replanner",...}
- REPORT       → {"type":"report","stage":"final_report",...}
- COMPLETE     → diagnosis_mode ? {"stage":"diagnosis_complete","diagnosis":{...}}
                  : {"type":"complete","response":...,"timed_out":...}
- ERROR        → {"type":"error","stage":"error","message":...}

原则：只增不改——新增事件类型在此处返回 None（跳过）即可，不影响既有消费者。
"""

from typing import Any

from app.agent.runtime.events import AgentEvent, EventType


def agent_event_to_legacy(
    event: AgentEvent, *, diagnosis_mode: bool = False
) -> dict[str, Any] | None:
    """把运行时事件翻译为 /api/aiops 的旧版 SSE dict

    Args:
        event: 运行时统一事件
        diagnosis_mode: True 时 complete 事件包装为 diagnosis 形状
            （/api/aiops 契约），False 时保留 execute() 原始形状

    Returns:
        dict | None: 旧版事件 dict；未知类型返回 None（调用方跳过）
    """
    payload = event.payload

    if event.type is EventType.PLAN_CREATED:
        plan = list(payload.get("plan", []))
        legacy: dict[str, Any] = {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan,
        }
        # 与旧 _format_planner_event 一致：仅在有内容时附带上下文字段
        if payload.get("kg_context"):
            legacy["kg_context"] = payload["kg_context"]
        if payload.get("query_intent"):
            legacy["query_intent"] = payload["query_intent"]
        if payload.get("diagnosis_events"):
            legacy["diagnosis_events"] = payload["diagnosis_events"]
        # P3 只增不改：结构化计划作为新增可选字段
        if payload.get("plan_structured"):
            legacy["plan_structured"] = payload["plan_structured"]
        return legacy

    if event.type is EventType.STEP_END:
        steps_done = int(payload.get("steps_done", 0))
        remaining = int(payload.get("remaining_steps", 0))
        return {
            "type": "step_complete",
            "stage": "step_executed",
            "message": f"步骤执行完成 ({steps_done}/{steps_done + remaining})",
            "current_step": payload.get("current_step", ""),
            "result_preview": payload.get("result_preview", ""),
            "remaining_steps": remaining,
        }

    if event.type is EventType.STEP_START:
        return {
            "type": "status",
            "stage": str(payload.get("stage", "executor")),
            "message": str(payload.get("message", "")),
        }

    if event.type is EventType.REPLAN:
        remaining = int(payload.get("remaining_steps", 0))
        return {
            "type": "status",
            "stage": "replanner",
            "message": f"评估完成，{'继续执行剩余步骤' if remaining else '准备生成最终响应'}",
            "remaining_steps": remaining,
        }

    if event.type is EventType.REPORT:
        return {
            "type": "report",
            "stage": "final_report",
            "message": "最终报告已生成",
            "report": payload.get("report", ""),
        }

    if event.type is EventType.COMPLETE:
        response = str(payload.get("response", ""))
        timed_out = bool(payload.get("timed_out", False))
        if diagnosis_mode:
            return {
                "type": "complete",
                "stage": "diagnosis_complete",
                "message": "诊断流程完成",
                "diagnosis": {"status": "completed", "report": response},
            }
        return {
            "type": "complete",
            "stage": "complete",
            "message": ("任务执行完成" if not timed_out else "任务超时，已生成部分报告"),
            "response": response,
            "timed_out": timed_out,
        }

    if event.type is EventType.ERROR:
        return {
            "type": "error",
            "stage": "error",
            "message": str(payload.get("message", "")),
        }

    # TOKEN / TOOL_* 等新事件类型：/api/aiops 旧契约不含，跳过
    return None
