"""统一 Agent 运行时层

三种范式共用同一事件协议（AgentEvent / EventType）与抽象接口（AgentRuntime）：

- ReActRuntime          — create_agent 思考-行动循环（对话式诊断）
- PlanExecuteRuntime    — Plan-Execute-Replan 工作流（复杂任务分解）
- ParallelRuntime       — 多专业 Agent 并行 + 汇总交叉验证

服务门面（app/services/*）与 API 层通过 run() 的 AsyncIterator[AgentEvent]
消费执行过程，SSE 契约由 app/api/event_translator.py 统一翻译。

P5：子模块改为惰性导出。本包 __init__ 曾急切拉起全部 runtime 子模块，
而 aiops 节点（executor/planner/replanner）需要在导入期引用
runtime.toolsets 工具清单——急切初始化会形成
aiops ⇄ runtime.plan_execute_runtime 的循环导入。
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 惰性导出的静态面：mypy 经此解析符号，运行时经下方 __getattr__ 加载
    from app.agent.runtime.base import AgentRuntime, RuntimeRegistry, default_registry
    from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType
    from app.agent.runtime.llm_factory import TieredLLM, tiered_llm
    from app.agent.runtime.parallel_runtime import (
        AgentFinding,
        DiagnosisResult,
        ParallelRuntime,
        run_parallel_diagnosis,
    )
    from app.agent.runtime.plan_execute_runtime import PlanExecuteRuntime
    from app.agent.runtime.react_runtime import ReActRuntime

_RUNTIME_EXPORTS: dict[str, str] = {
    "AgentRuntime": "app.agent.runtime.base",
    "RuntimeRegistry": "app.agent.runtime.base",
    "default_registry": "app.agent.runtime.base",
    "AgentEvent": "app.agent.runtime.events",
    "AgentEventEmitter": "app.agent.runtime.events",
    "EventType": "app.agent.runtime.events",
    "TieredLLM": "app.agent.runtime.llm_factory",
    "tiered_llm": "app.agent.runtime.llm_factory",
    "ParallelRuntime": "app.agent.runtime.parallel_runtime",
    "run_parallel_diagnosis": "app.agent.runtime.parallel_runtime",
    "DiagnosisResult": "app.agent.runtime.parallel_runtime",
    "AgentFinding": "app.agent.runtime.parallel_runtime",
    "PlanExecuteRuntime": "app.agent.runtime.plan_execute_runtime",
    "ReActRuntime": "app.agent.runtime.react_runtime",
}

__all__ = [
    "AgentEvent",
    "AgentEventEmitter",
    "AgentFinding",
    "AgentRuntime",
    "DiagnosisResult",
    "EventType",
    "ParallelRuntime",
    "PlanExecuteRuntime",
    "ReActRuntime",
    "RuntimeRegistry",
    "TieredLLM",
    "default_registry",
    "run_parallel_diagnosis",
    "tiered_llm",
]


def __getattr__(name: str) -> Any:
    """PEP 562 惰性导出：首次访问时加载对应子模块并缓存到包命名空间"""
    module_path = _RUNTIME_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr: Any = getattr(importlib.import_module(module_path), name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted({*globals(), *_RUNTIME_EXPORTS})
