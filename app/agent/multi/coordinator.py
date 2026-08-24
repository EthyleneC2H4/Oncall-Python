"""多 Agent 并行诊断 — Coordinator 兼容门面

实现已迁移至 app.agent.runtime.parallel_runtime（ParallelRuntime，
增量事件流版本）。本模块仅保留原公开符号的再导出与降级阶梯，
app/agent/multi/__init__.py 与 app/api/multi_diag.py 的既有导入不受影响。

架构（不变）：
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

from loguru import logger

from app.agent.runtime.parallel_runtime import (
    AgentFinding,
    DiagnosisResult,
    ParallelRuntime,
    run_parallel_diagnosis,
)
from app.core.degradation import DegradationLevel

__all__ = [
    "AgentFinding",
    "DiagnosisResult",
    "ParallelRuntime",
    "run_diagnosis_with_degradation",
    "run_parallel_diagnosis",
]


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
