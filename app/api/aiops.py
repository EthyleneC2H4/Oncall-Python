"""
AIOps 智能运维接口
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.api.event_translator import agent_event_to_legacy
from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service
from app.services.pending_actions import ActionStatus, get_pending_action_store
from app.tools.guard import decide_action, execute_approved

router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 自动获取当前系统的活动告警
    - 使用 Plan-Execute-Replan 模式进行智能诊断
    - 流式返回诊断过程和结果

    **SSE 事件类型：**

    1. `status` - 状态更新
       ```json
       {
         "type": "status",
         "stage": "fetching_alerts",
         "message": "正在获取系统告警信息..."
       }
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {
         "type": "plan",
         "stage": "plan_created",
         "message": "诊断计划已制定，共 6 个步骤",
         "target_alert": {...},
         "plan": ["步骤1: ...", "步骤2: ..."]
       }
       ```

    3. `step_complete` - 步骤执行完成
       ```json
       {
         "type": "step_complete",
         "stage": "step_executed",
         "message": "步骤执行完成 (2/6)",
         "current_step": "查询系统日志",
         "result_preview": "...",
         "remaining_steps": 4
       }
       ```

    4. `report` - 最终诊断报告
       ```json
       {
         "type": "report",
         "stage": "final_report",
         "message": "最终诊断报告已生成",
         "report": "# 故障诊断报告\\n...",
         "evidence": {...}
       }
       ```

    5. `complete` - 诊断完成
       ```json
       {
         "type": "complete",
         "stage": "diagnosis_complete",
         "message": "诊断流程完成",
         "diagnosis": {...}
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "诊断过程发生错误: ..."
       }
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    **前端使用示例：**
    ```javascript
    const eventSource = new EventSource('/api/aiops');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'plan') {
        console.log('诊断计划:', data.plan);
      } else if (data.type === 'step_complete') {
        console.log('步骤完成:', data.current_step);
      } else if (data.type === 'report') {
        console.log('最终报告:', data.report);
      } else if (data.type === 'complete') {
        console.log('诊断完成');
        eventSource.close();
      }
    };
    ```

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        try:
            async for runtime_event in aiops_service.diagnose(session_id=session_id):
                # AgentEvent → 旧版 SSE dict 契约（golden 快照钉死，只增不改）
                event = agent_event_to_legacy(runtime_event, diagnosis_mode=True)
                if event is None:
                    continue

                # 发送事件
                yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps(
                    {"type": "error", "stage": "exception", "message": f"诊断异常: {str(e)}"},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


# ──────────────── 高风险动作审批（P3 强制确认门） ────────────────


def _action_to_dict(action) -> dict:  # noqa: ANN001 - PendingAction 模型避免循环导入
    return {
        "action_id": action.action_id,
        "tool_name": action.tool_name,
        "args": action.args,
        "reason": action.reason,
        "session_id": action.session_id,
        "status": action.status.value,
        "created_at": action.created_at,
        "decided_at": action.decided_at,
        "result_preview": action.result_preview,
    }


@router.get("/actions/pending")
async def list_pending_actions():
    """列出待人工裁决的高风险动作"""
    store = get_pending_action_store()
    pending = await asyncio.to_thread(store.list_pending)
    return {"code": 200, "data": {"total": len(pending), "actions": [_action_to_dict(a) for a in pending]}}


@router.post("/actions/{action_id}/approve")
async def approve_action(action_id: str):
    """批准并补执行一个高风险动作

    流程：pending → approved → 原子认领(executed) → 按登记的参数补执行 → 回填结果预览。
    重复/并发 approve 因认领恰好一次语义而只会真正执行一次。
    """
    logger.info(f"审批请求: approve {action_id}")
    store = get_pending_action_store()
    decided = await asyncio.to_thread(decide_action, action_id, ActionStatus.APPROVED)
    if decided is None:
        raise HTTPException(status_code=404, detail=f"待审动作不存在: {action_id}")
    if decided.status is not ActionStatus.APPROVED:
        # 已裁决/已过期/已执行：返回现状态，不重复执行
        return {"code": 200, "data": {"action": _action_to_dict(decided), "executed": False}}

    result = await execute_approved(decided)
    refreshed = await asyncio.to_thread(store.get, action_id)
    return {
        "code": 200,
        "data": {
            "executed": result.ok,
            "result_preview": (result.value if result.ok else "")[:500],
            "error": result.error if not result.ok else "",
            "action": _action_to_dict(refreshed) if refreshed else None,
        },
    }


@router.post("/actions/{action_id}/reject")
async def reject_action(action_id: str):
    """拒绝一个高风险动作（终态，不执行）"""
    logger.info(f"审批请求: reject {action_id}")
    decided = await asyncio.to_thread(decide_action, action_id, ActionStatus.REJECTED)
    if decided is None:
        raise HTTPException(status_code=404, detail=f"待审动作不存在: {action_id}")
    return {"code": 200, "data": {"action": _action_to_dict(decided), "executed": False}}
