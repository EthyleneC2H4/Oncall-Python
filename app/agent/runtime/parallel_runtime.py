"""并行诊断运行时 — 三专业 Agent 并行执行，增量产出事件流

从 coordinator.py 提取编排逻辑（AgentFinding / DiagnosisResult 模型与
_run_agent_safe 一并迁移，coordinator.py 保留为兼容门面）。

与旧实现「gather 全部完成后一次性返回」不同，本运行时用队列边执行边发事件：
每个 Agent 启动即发 STEP_START，完成即发 STEP_END，
前端可在综合报告生成前就看到各专家的进度。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from app.agent.multi.specialists import (
    KnowledgeRetrieverAgent,
    LogAnalystAgent,
    MetricInspectorAgent,
)
from app.agent.multi.synthesizer import Synthesizer
from app.agent.runtime.base import AgentRuntime, default_registry
from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType
from app.core.degradation import DegradationLevel


class AgentFinding(BaseModel):
    """单个 Agent 的发现"""

    agent_name: str
    findings: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_ms: float = 0.0
    error: str = ""


class DiagnosisResult(BaseModel):
    """并行诊断汇总结果"""

    alert_input: str
    agent_findings: list[AgentFinding]
    synthesized_report: str
    total_duration_ms: float
    agents_succeeded: int
    agents_failed: int
    degradation_level: str = DegradationLevel.NONE.value


async def _run_agent_safe(agent: Any, alert_input: str) -> AgentFinding:
    """安全执行单个 Agent（异常不影响其他 Agent）"""
    start = time.time()
    try:
        result = await agent.analyze(alert_input)
        duration = (time.time() - start) * 1000
        return AgentFinding(
            agent_name=agent.name,
            findings=result,
            confidence=agent.confidence,
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"Agent {agent.name} 执行失败: {e}")
        return AgentFinding(
            agent_name=agent.name,
            findings="",
            confidence=0.0,
            duration_ms=duration,
            error=str(e),
        )


class ParallelRuntime(AgentRuntime):
    """并行诊断运行时：N 个专业 Agent 并行 + Synthesizer 汇总交叉验证"""

    name = "parallel"

    def __init__(self, agents: list | None = None, synthesizer: Synthesizer | None = None):
        self.agents = (
            agents
            if agents is not None
            else [
                LogAnalystAgent(),
                MetricInspectorAgent(),
                KnowledgeRetrieverAgent(),
            ]
        )
        self.synthesizer = synthesizer or Synthesizer()
        default_registry.register(self)

    async def run(self, task: str, session_id: str = "default") -> AsyncIterator[AgentEvent]:
        """并行执行所有专家 Agent，增量产出 STEP_START / STEP_END / REPORT / COMPLETE"""
        emitter = AgentEventEmitter(session_id=session_id)

        logger.info(f"=== 多 Agent 并行诊断启动（会话 {session_id}）===")
        start_time = time.time()

        # 队列：worker 把生命周期事件推给消费侧，实现真正的增量流
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _worker(agent: Any) -> None:
            await queue.put(("start", agent))
            finding = await _run_agent_safe(agent, task)
            await queue.put(("end", finding))

        worker_tasks = [asyncio.create_task(_worker(a)) for a in self.agents]

        findings: list[AgentFinding] = []
        try:
            remaining = len(self.agents) * 2  # 每个 Agent 一个 start + 一个 end
            while remaining > 0:
                kind, item = await queue.get()
                remaining -= 1
                if kind == "start":
                    yield emitter.emit(
                        EventType.STEP_START,
                        stage="agent_started",
                        agent=item.name,
                        message=f"{item.name} 开始分析",
                    )
                else:
                    findings.append(item)
                    succeeded = not item.error
                    yield emitter.emit(
                        EventType.STEP_END,
                        stage="agent_result",
                        agent=item.agent_name,
                        finding=item.model_dump(),
                        message=f"{item.agent_name} 分析完成"
                        if succeeded
                        else f"{item.agent_name} 执行失败",
                    )

            # 汇总交叉验证
            report = await self.synthesizer.synthesize(task, findings)

            succeeded_count = sum(1 for f in findings if not f.error)
            failed_count = len(findings) - succeeded_count
            total_ms = (time.time() - start_time) * 1000

            yield emitter.emit(EventType.REPORT, report=report)

            yield emitter.emit(
                EventType.COMPLETE,
                message="多 Agent 诊断完成",
                stats={
                    "total_duration_ms": total_ms,
                    "agents_succeeded": succeeded_count,
                    "agents_failed": failed_count,
                },
                degradation_level=DegradationLevel.NONE.value,
            )
            logger.info(f"=== 多 Agent 并行诊断完成，耗时 {total_ms:.0f}ms ===")

        finally:
            # 消费方提前断开时回收仍在执行的 worker
            for t in worker_tasks:
                if not t.done():
                    t.cancel()

    def snapshot(self, session_id: str) -> dict:
        """并行诊断为无状态单次执行，无会话快照"""
        return {"agents": [a.name for a in self.agents]}

    def reset(self, session_id: str) -> bool:
        """无会话状态可清空"""
        return True


async def run_parallel_diagnosis(
    alert_input: str, runtime: ParallelRuntime | None = None
) -> DiagnosisResult:
    """运行多 Agent 并行诊断（兼容入口：消费事件流汇总为 DiagnosisResult）

    Args:
        alert_input: 告警描述/诊断任务
        runtime: 可注入的运行时实例（测试用），默认新建

    Returns:
        DiagnosisResult: 汇总诊断结果
    """
    rt = runtime or ParallelRuntime()

    findings: list[AgentFinding] = []
    report = ""
    start_time = time.time()

    async for event in rt.run(alert_input, session_id="coordinator"):
        if event.type is EventType.STEP_END:
            findings.append(AgentFinding(**event.payload["finding"]))
        elif event.type is EventType.REPORT:
            report = str(event.payload.get("report", ""))

    succeeded = sum(1 for f in findings if not f.error)
    failed = len(findings) - succeeded

    return DiagnosisResult(
        alert_input=alert_input,
        agent_findings=findings,
        synthesized_report=report,
        total_duration_ms=(time.time() - start_time) * 1000,
        agents_succeeded=succeeded,
        agents_failed=failed,
    )
