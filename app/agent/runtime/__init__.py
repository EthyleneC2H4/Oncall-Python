"""统一 Agent 运行时层

三种范式共用同一事件协议（AgentEvent / EventType）与抽象接口（AgentRuntime）：

- ReActRuntime          — create_agent 思考-行动循环（对话式诊断）
- PlanExecuteRuntime    — Plan-Execute-Replan 工作流（复杂任务分解）
- ParallelRuntime       — 多专业 Agent 并行 + 汇总交叉验证

服务门面（app/services/*）与 API 层通过 run() 的 AsyncIterator[AgentEvent]
消费执行过程，SSE 契约由 app/api/event_translator.py 统一翻译。
"""

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
