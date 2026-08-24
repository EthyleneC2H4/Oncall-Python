"""Agent 运行时抽象基类与注册表

参照 Cortex 的 Runtime 抽象：所有范式（ReAct / Plan-Execute / 并行诊断）
实现同一接口，向上层（服务门面 / API）暴露统一的
「run -> AsyncIterator[AgentEvent]」事件流。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.agent.runtime.events import AgentEvent


class AgentRuntime(ABC):
    """Agent 运行时抽象基类"""

    #: 运行时名称（注册表主键），子类必须覆盖
    name: str = "base"

    @abstractmethod
    def run(self, task: str, session_id: str = "default") -> AsyncIterator[AgentEvent]:
        """执行一次完整任务，增量产出结构化事件

        约定：正常结束以 COMPLETE 事件收尾，异常以 ERROR 事件收尾；
        消费方在收到终止事件后可安全断开。
        """
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, session_id: str) -> dict:
        """读取指定会话的当前状态快照（如消息历史）"""
        raise NotImplementedError

    @abstractmethod
    def reset(self, session_id: str) -> bool:
        """清空指定会话的状态；返回是否成功"""
        raise NotImplementedError


class RuntimeRegistry:
    """运行时注册表：按 name 注册 / 查找运行时实例"""

    def __init__(self) -> None:
        self._runtimes: dict[str, AgentRuntime] = {}

    def register(self, runtime: AgentRuntime) -> None:
        """注册运行时（同名覆盖，后注册者生效）"""
        self._runtimes[runtime.name] = runtime

    def get(self, name: str) -> AgentRuntime | None:
        """按名称查找运行时；不存在返回 None"""
        return self._runtimes.get(name)

    def names(self) -> list[str]:
        """已注册的运行时名称列表"""
        return list(self._runtimes.keys())


# 默认全局注册表：各运行时构造时自动注册，便于内省与统一调度
default_registry = RuntimeRegistry()
