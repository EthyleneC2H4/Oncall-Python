"""多 Agent 并行诊断接口"""

import json

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent.multi.coordinator import run_diagnosis_with_degradation

router = APIRouter()


class MultiDiagRequest(BaseModel):
    """多 Agent 诊断请求"""

    alert_input: str = Field(description="告警描述或诊断任务")
    session_id: str = Field(default="default", description="会话ID")


@router.post("/multi-diagnose")
async def multi_agent_diagnose(request: MultiDiagRequest):
    """多 Agent 并行诊断接口（SSE 流式）

    使用 3 个专业 Agent（日志分析、指标检查、知识检索）并行分析，
    Synthesizer 交叉验证后生成综合报告。

    SSE 事件：
    - agent_start: Agent 开始执行
    - agent_complete: 单个 Agent 完成
    - synthesis: 综合报告生成
    - complete: 全部完成
    """
    logger.info(f"多 Agent 诊断请求: {request.alert_input}")

    async def event_generator():
        try:
            # 发送启动事件
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "status",
                        "stage": "multi_agent_start",
                        "message": "多 Agent 并行诊断启动（日志分析 + 指标检查 + 知识检索）",
                    },
                    ensure_ascii=False,
                ),
            }

            # 执行并行诊断（带降级保护）
            result = await run_diagnosis_with_degradation(request.alert_input)

            # 发送各 Agent 结果
            for finding in result.agent_findings:
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "type": "agent_complete",
                            "stage": "agent_result",
                            "agent_name": finding.agent_name,
                            "confidence": finding.confidence,
                            "duration_ms": finding.duration_ms,
                            "error": finding.error,
                            "findings_preview": finding.findings[:300] if finding.findings else "",
                        },
                        ensure_ascii=False,
                    ),
                }

            # 发送综合报告
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "report",
                        "stage": "synthesis_complete",
                        "message": "综合诊断报告已生成",
                        "report": result.synthesized_report,
                        "stats": {
                            "total_duration_ms": result.total_duration_ms,
                            "agents_succeeded": result.agents_succeeded,
                            "agents_failed": result.agents_failed,
                        },
                    },
                    ensure_ascii=False,
                ),
            }

            # 降级状态事件（如果有降级）
            if result.degradation_level != "none":
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "type": "degradation_status",
                            "is_degraded": True,
                            "level": result.degradation_level,
                        },
                        ensure_ascii=False,
                    ),
                }

            # 完成事件
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "complete",
                        "stage": "multi_diagnosis_complete",
                        "message": "多 Agent 诊断完成",
                        "degradation_level": result.degradation_level,
                    },
                    ensure_ascii=False,
                ),
            }

        except Exception as e:
            logger.error(f"多 Agent 诊断失败: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "message": f"多 Agent 诊断失败: {str(e)}",
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())
