"""多 Agent 并行诊断模块

实现 Coordinator → 专业 Agent 并行 → Synthesizer 架构。
"""

from app.agent.multi.coordinator import run_parallel_diagnosis

__all__ = ["run_parallel_diagnosis"]
