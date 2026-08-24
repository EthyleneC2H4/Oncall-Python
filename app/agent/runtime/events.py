"""统一结构化事件协议

所有 Agent 运行时（ReAct / Plan-Execute / 并行诊断）通过 AgentEvent 汇报执行过程，
API 层再将 AgentEvent 翻译为各端点的 SSE 契约（旧事件 dict 由翻译器钉死，见
app/api/event_translator.py）。

事件类型只增不改：新增 EventType 成员不影响既有消费者。
"""

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Agent 运行时事件类型"""

    TOKEN = "token"  # LLM 增量文本片段
    TOOL_START = "tool_start"  # 工具调用开始
    TOOL_END = "tool_end"  # 工具调用结束
    STEP_START = "step_start"  # 步骤/子任务开始（并行诊断中的单个 Agent 等）
    STEP_END = "step_end"  # 步骤/子任务结束
    PLAN_CREATED = "plan_created"  # 计划制定完成
    REPLAN = "replan"  # 重新规划决策
    REPORT = "report"  # 最终报告生成
    COMPLETE = "complete"  # 整次 run 正常结束（终止事件）
    ERROR = "error"  # 整次 run 异常结束（终止事件）

    @classmethod
    def terminal_types(cls) -> set["EventType"]:
        """终止事件类型集合（COMPLETE / ERROR），流消费方可据此断开"""
        return {cls.COMPLETE, cls.ERROR}


class AgentEvent(BaseModel):
    """运行时统一事件

    seq 在单次 run 内自增；run_id 标识一次完整执行；
    payload 为事件类型的专属字段集合（如 TOKEN 的 text / TOOL_END 的 result_preview）。
    """

    seq: int = Field(ge=1, description="run 内自增序号，从 1 开始")
    session_id: str = Field(description="会话 ID")
    run_id: str = Field(description="本次执行的唯一标识")
    type: EventType = Field(description="事件类型")
    timestamp: float = Field(description="Unix 时间戳（秒）")
    payload: dict[str, Any] = Field(default_factory=dict, description="事件负载")


class AgentEventEmitter:
    """单次 run 的事件发射器：自动维护 seq / session_id / run_id / timestamp

    用法：
        emitter = AgentEventEmitter(session_id="s1")
        event = emitter.emit(EventType.TOKEN, text="你好", node="agent")
    """

    def __init__(self, session_id: str, run_id: str | None = None):
        self.session_id = session_id
        self.run_id = run_id or uuid.uuid4().hex
        self._seq = 0

    def emit(self, type: EventType, **payload: Any) -> AgentEvent:
        """构造下一个事件（seq 自增，时间戳取当前时刻）"""
        self._seq += 1
        return AgentEvent(
            seq=self._seq,
            session_id=self.session_id,
            run_id=self.run_id,
            type=type,
            timestamp=time.time(),
            payload=payload,
        )

    @property
    def seq(self) -> int:
        """已发射的事件数量（最后一个事件的 seq）"""
        return self._seq
