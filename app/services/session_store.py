"""会话读写服务（P5 自 runtime 层抽出）

LangGraph checkpointer 之上的会话历史统一读写视图：
- 读：MemorySaver 检查点 → channel_values.messages → 过滤系统消息后的
  [{"role", "content", "timestamp"}] 列表（原 react_runtime.snapshot 的
  转换逻辑，含 CheckpointTuple/普通元组双形态收窄）
- 清：delete_thread 的两种容错语义——严格版返回成败、尽力版只告警
  （Plan-Execute 每次 run 前的跨运行残留清理用后者）

工作流级状态读取（graph.get_state 的 {"values": ...} 视图）与具体图
绑定，仍留在 PlanExecuteRuntime.snapshot，不入本服务。
"""

from datetime import datetime
from typing import Any

from loguru import logger


class SessionStore:
    """基于 checkpointer 的会话历史读写（一个 checkpointer 一个实例）"""

    def __init__(self, checkpointer: Any) -> None:
        self.checkpointer = checkpointer

    def read_messages(self, session_id: str) -> list[dict[str, Any]]:
        """读取会话消息历史（用户/助手轮次，不含系统提示词）"""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.runnables import RunnableConfig

        try:
            run_config = RunnableConfig(configurable={"thread_id": session_id})
            checkpoint_result = self.checkpointer.get(run_config)

            if not checkpoint_result:
                return []

            # 形态收窄（评审修复）：langgraph 的 get() 返回的是 Checkpoint
            # 本体——裸 dict，顶层即 channel_values/channel_versions 等键；
            # 其余两分支兼容属性对象与旧式元组包装（测试桩/历史后端）
            if isinstance(checkpoint_result, dict):
                checkpoint_data: Any = checkpoint_result
            elif isinstance(checkpoint_result, tuple):
                checkpoint_data = checkpoint_result[0] if len(checkpoint_result) > 0 else {}
            else:
                checkpoint_data = getattr(checkpoint_result, "checkpoint", {}) or {}

            checkpoint_dict: dict = dict(checkpoint_data) if isinstance(checkpoint_data, dict) else {}
            channel_values = checkpoint_dict.get("channel_values") or {}
            messages = (
                list(channel_values.get("messages") or [])
                if isinstance(channel_values, dict)
                else []
            )

            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                timestamp = getattr(msg, "timestamp", None)
                if not timestamp:
                    timestamp = datetime.now().isoformat()
                history.append({"role": role, "content": content, "timestamp": timestamp})

            return history
        except Exception as e:
            logger.error(f"读取会话快照失败: {session_id}, 错误: {e}")
            return []

    def clear(self, session_id: str) -> bool:
        """清空指定会话线程。失败记录错误并返回 False，由调用方决定是否阻断"""
        try:
            self.checkpointer.delete_thread(session_id)
            return True
        except Exception as e:
            logger.error(f"清空会话失败: {session_id}, 错误: {e}")
            return False

    def clear_best_effort(self, session_id: str) -> None:
        """尽力清空：任务开始前的残留清理，失败只告警绝不阻断本次执行"""
        try:
            self.checkpointer.delete_thread(session_id)
        except Exception as e:  # noqa: BLE001 - 清理失败不阻断执行
            logger.warning(f"清空会话检查点失败（忽略）: {e}")
