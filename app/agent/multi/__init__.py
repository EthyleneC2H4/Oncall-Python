"""多 Agent 并行诊断模块

实现 Coordinator → 专业 Agent 并行 → Synthesizer 架构。

实现位于 app.agent.runtime.parallel_runtime（coordinator.py 为兼容门面）。
为避免 parallel_runtime ↔ multi 的循环导入，此处不做顶层再导出，
按需经 __getattr__ 惰性转发（`from app.agent.multi import run_parallel_diagnosis` 仍可用）。
"""

from typing import Any

_LAZY_EXPORTS = {
    "run_parallel_diagnosis": "app.agent.multi.coordinator",
    "run_diagnosis_with_degradation": "app.agent.multi.coordinator",
    "AgentFinding": "app.agent.multi.coordinator",
    "DiagnosisResult": "app.agent.multi.coordinator",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)
