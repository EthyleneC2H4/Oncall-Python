"""ReAct 运行时 — 包装 create_agent，产出统一结构化事件流

从 rag_agent_service.py 提取：Agent 构造、MCP 工具装载、会话检查点管理，
以及基于 stream_mode=["messages","updates"] 的双通道流式：

- messages 通道 → TOKEN 事件（增量文本）
- updates 通道 → TOOL_START / TOOL_END 事件（节点级状态增量）

原实现只消费 messages 通道，前端看不到工具调用过程；
双通道后 TOOL 事件为纯增量（SSE 契约「只增不改」）。
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.agent.runtime.base import AgentRuntime, default_registry
from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType
from app.config import config
from app.core.llm_factory import LLMFactory

# 文本 token 所在的 AIMessageChunk 类型名（与旧实现保持一致的白名单）
_TOKEN_MESSAGE_TYPES = ("AIMessage", "AIMessageChunk")

# 工具结果预览截断长度
_TOOL_PREVIEW_CHARS = 300


class ReActRuntime(AgentRuntime):
    """ReAct 运行时：思考 → 工具调用 → 观察 循环"""

    name = "react"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        tools: list | None = None,
        system_prompt: str = "",
    ):
        """初始化运行时（惰性：MCP 工具与 Agent 在首次 run 时装载）

        Args:
            model_name: OpenRouter 模型 slug，默认 config.rag_model
            temperature: 采样温度
            streaming: 是否流式（影响 LLM 客户端构造）
            tools: 基础工具列表，默认运维诊断四件套
            system_prompt: 系统提示词（由服务门面从 Prompt 管理器构建后注入）
        """
        self.model_name = model_name or config.rag_model
        self.temperature = temperature
        self.streaming = streaming
        self.system_prompt = system_prompt

        if tools is not None:
            self.tools = list(tools)
        else:
            # 默认基础工具（含知识图谱工具）；延迟导入避免循环依赖
            from app.tools import (
                get_current_time,
                predict_alert_cascade,
                query_alert_graph,
                retrieve_knowledge,
            )

            self.tools = [
                retrieve_knowledge,
                get_current_time,
                query_alert_graph,
                predict_alert_cascade,
            ]

        self.mcp_tools: list = []

        # 会话检查点（MemorySaver）
        self.checkpointer = MemorySaver()

        self.agent: Any = None
        self._initialized = False

        default_registry.register(self)
        logger.info(
            f"ReActRuntime 初始化完成, model={self.model_name}, streaming={streaming}"
        )

    async def ensure_ready(self) -> None:
        """异步初始化 Agent（包括 MCP 工具；幂等）"""
        if self._initialized:
            return

        # MCP 工具列表带短 TTL 缓存（见 mcp_client.get_mcp_tools）
        self.mcp_tools = await get_mcp_tools()
        logger.info(f"成功加载 {len(self.mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools

        model = LLMFactory.create_chat_model(
            model=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming,
        )

        self.agent = create_agent(
            model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )
        self._initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_input(self, task: str) -> dict[str, Any]:
        """构建 Agent 输入消息列表"""
        messages: list = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=task))
        return {"messages": messages}

    async def run(self, task: str, session_id: str = "default") -> AsyncIterator[AgentEvent]:
        """流式执行一次查询，产出 TOKEN / TOOL_START / TOOL_END / COMPLETE|ERROR"""
        emitter = AgentEventEmitter(session_id=session_id)

        try:
            await self.ensure_ready()
            assert self.agent is not None, "Agent 未初始化"

            logger.info(f"[会话 {session_id}] ReActRuntime 收到任务: {task}")

            agent_input = self._build_input(task)
            run_config = {"configurable": {"thread_id": session_id}}

            answer_parts: list[str] = []
            # tool_call id → {"name","args"}，用于 TOOL_END 时回填工具名
            pending_calls: dict[str, dict[str, Any]] = {}

            async for mode, chunk in self.agent.astream(
                input=agent_input,
                config=run_config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    # (消息块, 元数据)：AIMessageChunk 的文本块即 token
                    token, metadata = chunk
                    node_name = (
                        metadata.get("langgraph_node", "unknown")
                        if isinstance(metadata, dict)
                        else "unknown"
                    )
                    if type(token).__name__ not in _TOKEN_MESSAGE_TYPES:
                        continue
                    for event in self._emit_token_events(token, node_name, emitter):
                        answer_parts.append(str(event.payload.get("text", "")))
                        yield event
                else:
                    # updates：{node_name: state_delta}
                    for event in self._emit_update_events(chunk, emitter, pending_calls):
                        yield event

            logger.info(f"[会话 {session_id}] ReActRuntime 任务完成")
            yield emitter.emit(EventType.COMPLETE, message="查询完成", answer="".join(answer_parts))

        except Exception as e:
            # ERROR 即终止事件：流正常收尾，不再向消费方抛异常
            # （非流式门面如需异常语义，自行根据 ERROR 事件转换）
            logger.error(f"[会话 {session_id}] ReActRuntime 任务失败: {e}", exc_info=True)
            yield emitter.emit(EventType.ERROR, message=str(e))

    @staticmethod
    def _emit_token_events(
        token: Any, node_name: str, emitter: AgentEventEmitter
    ) -> list[AgentEvent]:
        """从 AIMessageChunk 提取文本块，转为 TOKEN 事件"""
        events: list[AgentEvent] = []
        content_blocks = getattr(token, "content_blocks", None)
        if not content_blocks or not isinstance(content_blocks, list):
            return events
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_content = block.get("text", "")
                if text_content:
                    events.append(emitter.emit(EventType.TOKEN, text=text_content, node=node_name))
        return events

    @classmethod
    def _emit_update_events(
        cls,
        update: dict[str, Any],
        emitter: AgentEventEmitter,
        pending_calls: dict[str, dict[str, Any]],
    ) -> list[AgentEvent]:
        """从节点状态增量提取 TOOL_START / TOOL_END 事件"""
        events: list[AgentEvent] = []
        for node_name, delta in update.items():
            messages = (delta or {}).get("messages") or []
            for msg in messages:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        call_id = str(tc.get("id") or tc.get("name", ""))
                        pending_calls[call_id] = {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        }
                        events.append(
                            emitter.emit(
                                EventType.TOOL_START,
                                tool=tc.get("name", ""),
                                args=tc.get("args", {}),
                                node=node_name,
                            )
                        )
                if isinstance(msg, ToolMessage):
                    call_id = str(getattr(msg, "tool_call_id", "") or "")
                    meta = pending_calls.get(call_id, {})
                    status = "error" if getattr(msg, "status", "") == "error" else "success"
                    events.append(
                        emitter.emit(
                            EventType.TOOL_END,
                            tool=msg.name or meta.get("name", ""),
                            status=status,
                            result_preview=str(msg.content)[:_TOOL_PREVIEW_CHARS],
                            node=node_name,
                        )
                    )
        return events

    def snapshot(self, session_id: str) -> dict:
        """读取会话消息历史（从 MemorySaver 检查点）"""
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig

        try:
            run_config = RunnableConfig(configurable={"thread_id": session_id})
            checkpoint_tuple = self.checkpointer.get(run_config)

            if not checkpoint_tuple:
                return {"messages": []}

            # CheckpointTuple 兼容普通元组形态时先做 isinstance 收窄
            if isinstance(checkpoint_tuple, tuple):
                checkpoint_data = checkpoint_tuple[0] if len(checkpoint_tuple) > 0 else {}
            else:
                checkpoint_data = getattr(checkpoint_tuple, "checkpoint", {}) or {}

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
                    from datetime import datetime

                    timestamp = datetime.now().isoformat()
                history.append({"role": role, "content": content, "timestamp": timestamp})

            return {"messages": history}
        except Exception as e:
            logger.error(f"读取会话快照失败: {session_id}, 错误: {e}")
            return {"messages": []}

    def reset(self, session_id: str) -> bool:
        """清空指定会话的检查点"""
        try:
            self.checkpointer.delete_thread(session_id)
            return True
        except Exception as e:
            logger.error(f"清空会话失败: {session_id}, 错误: {e}")
            return False
