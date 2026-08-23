"""多 Agent 并行诊断 - Coordinator 调度器

架构：
    ┌──────────────┐
    │  Coordinator │ ← 总调度（分析告警，分发任务）
    └──────┬───────┘
           │ 并行分发
    ┌──────┼──────────┐
    ↓      ↓          ↓
  Log    Metric    Knowledge
  Agent  Agent     Agent
    │      │          │
    ↓      ↓          ↓
    ┌─────────────────────┐
    │   Synthesizer       │
    │  汇总 → 交叉验证    │
    └─────────────────────┘
"""

import asyncio
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from app.agent.multi.specialists import (
    KnowledgeRetrieverAgent,
    LogAnalystAgent,
    MetricInspectorAgent,
)
from app.agent.multi.synthesizer import Synthesizer
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


async def run_parallel_diagnosis(alert_input: str) -> DiagnosisResult:
    """运行多 Agent 并行诊断

    三个专业 Agent 并行执行，Synthesizer 汇总交叉验证。

    Args:
        alert_input: 告警描述/诊断任务

    Returns:
        DiagnosisResult: 汇总诊断结果
    """
    logger.info("=== 多 Agent 并行诊断启动 ===")
    logger.info(f"输入: {alert_input}")
    start_time = time.time()

    # 初始化专业 Agent
    log_agent = LogAnalystAgent()
    metric_agent = MetricInspectorAgent()
    knowledge_agent = KnowledgeRetrieverAgent()

    # 并行执行三个 Agent
    tasks = [
        _run_agent_safe(log_agent, alert_input),
        _run_agent_safe(metric_agent, alert_input),
        _run_agent_safe(knowledge_agent, alert_input),
    ]

    findings = await asyncio.gather(*tasks)

    # 统计成功/失败
    succeeded = sum(1 for f in findings if not f.error)
    failed = sum(1 for f in findings if f.error)
    logger.info(f"Agent 执行完成: {succeeded} 成功, {failed} 失败")

    # Synthesizer 汇总
    synthesizer = Synthesizer()
    report = await synthesizer.synthesize(alert_input, findings)

    total_ms = (time.time() - start_time) * 1000

    result = DiagnosisResult(
        alert_input=alert_input,
        agent_findings=findings,
        synthesized_report=report,
        total_duration_ms=total_ms,
        agents_succeeded=succeeded,
        agents_failed=failed,
    )

    logger.info(f"=== 多 Agent 诊断完成，耗时 {total_ms:.0f}ms ===")
    return result


async def run_diagnosis_with_degradation(alert_input: str) -> DiagnosisResult:
    """带降级的诊断入口

    Level 0: 多 Agent 并行 (≥2 Agent 成功)
    Level 1: 多 Agent 部分成功 (仅 1 Agent 成功)
    Level 2: 直接 RAG 检索兜底

    Args:
        alert_input: 告警描述

    Returns:
        DiagnosisResult (含 degradation_level)
    """
    try:
        result = await asyncio.wait_for(
            run_parallel_diagnosis(alert_input),
            timeout=120,  # 2 分钟上限
        )
        if result.agents_succeeded >= 2:
            return result  # 正常

        # 仅 1 个 Agent 成功，标记降级
        result.degradation_level = DegradationLevel.SINGLE_AGENT.value
        logger.warning(f"多 Agent 仅 {result.agents_succeeded} 个成功，标记为降级")
        return result

    except TimeoutError:
        logger.error("多 Agent 诊断超时 (120s)")
    except Exception as e:
        logger.error(f"多 Agent 诊断异常: {e}")

    # 降级兜底：使用 RAG 检索
    from app.tools.knowledge_tool import retrieve_with_degradation

    try:
        ctx, docs, deg_level = await retrieve_with_degradation(alert_input)
        report = f"# 降级诊断结果\n\n多 Agent 诊断不可用，以下为 RAG 检索结果：\n\n{ctx}"
    except Exception:
        report = "诊断服务暂时不可用，请稍后重试。"
        deg_level = DegradationLevel.TEMPLATE

    return DiagnosisResult(
        alert_input=alert_input,
        agent_findings=[],
        synthesized_report=report,
        total_duration_ms=0.0,
        agents_succeeded=0,
        agents_failed=0,
        degradation_level=deg_level.value if hasattr(deg_level, "value") else str(deg_level),
    )


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
