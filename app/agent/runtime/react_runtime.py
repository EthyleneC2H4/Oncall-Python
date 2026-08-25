"""ReAct 运行时 — 包装 create_agent，产出统一结构化事件流

从 rag_agent_service.py 提取：Agent 构造、MCP 工具装载、会话检查点管理，
以及基于 stream_mode=["messages","updates"] 的双通道流式：

- messages 通道 → TOKEN 事件（增量文本）
- updates 通道 → TOOL_START / TOOL_END 事件（节点级状态增量）

原实现只消费 messages 通道，前端看不到工具调用过程；
双通道后 TOOL 事件为纯增量（SSE 契约「只增不改」）。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.agent.runtime.base import AgentRuntime, default_registry
from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType
from app.agent.runtime.middleware import TokenTrimMiddleware
from app.agent.runtime.session_mutex import session_mutex
from app.config import config
from app.core.llm_factory import LLMFactory
from app.services.session_store import SessionStore

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
            from app.agent.runtime.toolsets import local_toolkit

            self.tools = local_toolkit()

        self.mcp_tools: list = []
        # P5 AB：变体提示词 → (渲染文本, 独立编译图)（懒建；与主图共享 checkpointer）。
        # 缓存值携带渲染文本：块热加载导致文本变化时重建（评审修复——
        # 否则变体图永远持有构建期的提示词快照）
        self._variant_agents: dict[str, tuple[str, Any]] = {}
        # 每变体一把锁：并发首请求只编译一次图（评审修复）
        self._variant_locks: dict[str, asyncio.Lock] = {}

        # 会话检查点（MemorySaver）；会话读写视图见 _store 属性
        self.checkpointer = MemorySaver()
        # 检查点 LRU：thread → 最近活跃时间，超 checkpoint_max_threads 整体淘汰
        self._thread_last_access: dict[str, float] = {}

        # 会话互斥：同 session 并发 run 串行化，防检查点交错写入/串话
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 基线初始化锁：并发首请求只拉一次 MCP 工具、只编译一次图
        self._init_lock = asyncio.Lock()

        self.agent: Any = None
        self._initialized = False

        default_registry.register(self)
        logger.info(f"ReActRuntime 初始化完成, model={self.model_name}, streaming={streaming}")

    async def ensure_ready(self) -> None:
        """异步初始化 Agent（包括 MCP 工具；幂等 + 并发安全）

        锁内二次检查：并发首请求窗内只拉一次 MCP 工具、编译一次图——
        无锁的 check-then-act 会双重编译并竞写 self.agent/self.mcp_tools。
        """
        if self._initialized:
            return
        async with self._init_lock:
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

            # P2 修复：system_prompt 由 create_agent 在推理时统一注入（不落线程状态）。
            # 旧实现把 SystemMessage 塞进输入消息列表，会随轮次累积进检查点；
            # 且「最旧优先」的 token 裁剪会先砍掉携带最新记忆块的那条系统消息。
            agent_kwargs: dict[str, Any] = {
                "checkpointer": self.checkpointer,
                # P2: 按 token 预算裁剪历史（替换旧硬编码条数截断）
                "middleware": [TokenTrimMiddleware(max_tokens=config.context_history_budget)],
            }
            if self.system_prompt:
                agent_kwargs["system_prompt"] = self.system_prompt
            self.agent = create_agent(model, tools=all_tools, **agent_kwargs)
            self._initialized = True

            if all_tools:
                tool_names = [
                    tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools
                ]
                logger.info(f"可用工具列表: {', '.join(tool_names)}")

    async def _agent_for_variant(self, prompt_variant: str | None) -> tuple[Any, str]:
        """按变体取编译图，返回 (编译图, 实际生效的变体名)

        create_agent 在构造期绑定 system_prompt，变体因此需要独立编译图；
        checkpointer 与工具池共享，线程历史在基线与变体间连续。
        - 基线或回退（未登记变体/渲染失败）→ 主实例 + ""：
          归因以实际使用的图为准（评审修复——此前调用方按「请求的变体」
          预标记，未生效的变体也会被计入分组）
        - 缓存值携带渲染文本，块热加载后文本变化即重建图

        并发安全（评审修复）：每变体一把锁 + 锁内二次检查，
        首个并发请求窗内的多个请求只编译一次图。
        """
        if not prompt_variant:
            await self.ensure_ready()
            return self.agent, ""

        try:
            from app.core.prompt_manager import prompt_manager

            # 未登记变体先在解析层暴露：直接回退主实例，避免为一段
            # 与基线相同的正文重复编译一张图、并把未生效名计入归因
            resolved = prompt_manager.effective_variant("system_prompt", prompt_variant)
            if not resolved:
                logger.warning(f"变体 '{prompt_variant}' 未登记，回退基线图")
                await self.ensure_ready()
                return self.agent, ""
            variant_prompt = prompt_manager.render_composed("system_prompt", variant=prompt_variant)
        except Exception as e:
            logger.warning(f"变体 '{prompt_variant}' 提示词渲染失败，回退基线: {e}")
            await self.ensure_ready()
            return self.agent, ""

        async def _build() -> tuple[str, Any]:
            await self.ensure_ready()  # 确保工具池已装载
            model = LLMFactory.create_chat_model(
                model=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
            )
            agent_kwargs: dict[str, Any] = {
                "checkpointer": self.checkpointer,
                "middleware": [TokenTrimMiddleware(max_tokens=config.context_history_budget)],
            }
            if variant_prompt:
                agent_kwargs["system_prompt"] = variant_prompt
            agent = create_agent(model, tools=[*self.tools, *self.mcp_tools], **agent_kwargs)
            logger.info(f"Prompt 变体 '{prompt_variant}' 编译图已构建并缓存")
            return variant_prompt, agent

        # 快路径：文本一致直接命中
        cached = self._variant_agents.get(prompt_variant)
        if cached is not None and cached[0] == variant_prompt:
            return cached[1], prompt_variant

        lock = self._variant_locks.setdefault(prompt_variant, asyncio.Lock())
        async with lock:
            cached = self._variant_agents.get(prompt_variant)
            if cached is None or cached[0] != variant_prompt:
                self._variant_agents[prompt_variant] = await _build()
            return self._variant_agents[prompt_variant][1], prompt_variant

    def _build_input(self, task: str, *, memory_block: str = "") -> dict[str, Any]:
        """构建 Agent 输入消息列表

        Args:
            task: 用户任务
            memory_block: 记忆召回块（非空时作为本轮用户消息前缀注入；
                不再走 SystemMessage —— 那会随轮次累积进检查点状态，
                并被最旧优先的 token 裁剪优先砍掉携带最新记忆的消息）
        """
        content = f"[相关记忆]\n{memory_block}\n\n{task}" if memory_block else task
        return {"messages": [HumanMessage(content=content)]}

    async def _recall_memory(self, task: str) -> str:
        """召回与任务相关的长期记忆（失败安全，disabled 时返回空串）"""
        try:
            from app.core.context_engine import format_memory_block
            from app.services.memory import MemoryType, memory_service

            if not memory_service.enabled:
                return ""
            recalled = await memory_service.recall(
                task, types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
            )
            return format_memory_block(recalled)
        except Exception as e:
            logger.warning(f"记忆召回失败（忽略）: {e}")
            return ""

    async def _remember_turn(
        self, task: str, answer: str, tool_calls: int, session_id: str
    ) -> None:
        """把本轮交互写入情景记忆（失败安全）"""
        if not answer.strip():
            return
        try:
            from app.services.memory import memory_service

            if not memory_service.enabled:
                return
            importance = min(0.3 + 0.05 * tool_calls, 0.6)
            await memory_service.write_episodic(
                f"Q: {task}\nA: {answer[:500]}",
                session_id=session_id,
                importance=importance,
                metadata={"runtime": "react", "tool_calls": tool_calls},
            )
        except Exception as e:
            logger.warning(f"情景记忆写入失败（忽略）: {e}")

    async def run(
        self,
        task: str,
        session_id: str = "default",
        *,
        prompt_variant: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行一次查询，产出 TOKEN / TOOL_START / TOOL_END / COMPLETE|ERROR

        Args:
            task: 用户任务
            session_id: 会话 ID（thread_id）
            prompt_variant: 已解析生效的 Prompt 变体名（""/None 用基线图）
        """
        emitter = AgentEventEmitter(session_id=session_id)

        # 声明提前到 try 外：超时分支需要读取已收集的部分回答
        answer_parts: list[str] = []
        tool_event_count = 0

        # 会话互斥：同 session 并发 run 串行化，防检查点交错写入/串话
        async with session_mutex(self._session_locks, session_id):
            try:
                agent, used_variant = await self._agent_for_variant(prompt_variant)
                assert agent is not None, "Agent 未初始化"

                # AB 归因单一事实源（评审修复）：按实际生效的图记录，
                # 请求了但回退基线的情况计入 base 分组
                try:
                    from app.core.cost_tracker import cost_tracker

                    cost_tracker.mark_prompt_variant(used_variant, session_id=session_id)
                except Exception as e:  # noqa: BLE001 - 归因失败不影响主流程
                    logger.debug(f"Prompt 变体归因记录失败: {e}")

                logger.info(
                    f"[会话 {session_id}] ReActRuntime 收到任务: {task}"
                    + (f" (variant={prompt_variant})" if prompt_variant else "")
                )

                # 检查点 LRU 触发点：本轮线程记为活跃，超上限淘汰最久未活跃线程
                self._touch_thread(session_id)

                # P2: 轮前召回长期记忆注入 system prompt 尾部（经验复用闭环）
                memory_block = await self._recall_memory(task)
                agent_input = self._build_input(task, memory_block=memory_block)
                if memory_block:
                    logger.info(f"[会话 {session_id}] 注入记忆召回块 ({len(memory_block)} 字符)")

                run_config = {"configurable": {"thread_id": session_id}}

                # tool_call id → {"name","args"}，用于 TOOL_END 时回填工具名
                pending_calls: dict[str, dict[str, Any]] = {}

                # 工作流整体 deadline：三个运行时唯独此前的主聊天路径没有，
                # OpenRouter 停滞时 SSE 流永久挂起还占着并发槽（与 plan_execute 对齐）
                async with asyncio.timeout(config.workflow_timeout_seconds):
                    async for mode, chunk in agent.astream(
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
                            new_events = self._emit_update_events(chunk, emitter, pending_calls)
                            tool_event_count += sum(
                                1 for e in new_events if e.type is EventType.TOOL_START
                            )
                            for event in new_events:
                                yield event

                final_answer = "".join(answer_parts)
                # P2: 轮后写入情景记忆（下次同类问题可被召回）
                await self._remember_turn(task, final_answer, tool_event_count, session_id)

                logger.info(f"[会话 {session_id}] ReActRuntime 任务完成")
                # 变体生效时随 COMPLETE 下发实际变体名（SSE 只增不改：
                # 基线运行不携带该字段，消费方按缺省处理）
                complete_kwargs: dict[str, Any] = (
                    {"prompt_variant": used_variant} if used_variant else {}
                )
                yield emitter.emit(
                    EventType.COMPLETE,
                    message="查询完成",
                    answer=final_answer,
                    **complete_kwargs,
                )

            except TimeoutError:
                # 超时优雅截断：已产出的部分回答照常下发（与 plan_execute 的
                # 部分报告哲学一致），前端能正常收尾而不是等一个 ERROR
                partial_answer = "".join(answer_parts)
                logger.warning(
                    f"[会话 {session_id}] 工作流超时 "
                    f"({config.workflow_timeout_seconds}s)，返回部分回答"
                )
                yield emitter.emit(
                    EventType.COMPLETE,
                    message="已达到工作流超时上限，返回部分回答",
                    answer=partial_answer,
                    timed_out=True,
                )
            except Exception as e:
                # ERROR 即终止事件：流正常收尾，不再向消费方抛异常
                # （非流式门面如需异常语义，自行根据 ERROR 事件转换）
                logger.error(f"[会话 {session_id}] ReActRuntime 任务失败: {e}", exc_info=True)
                yield emitter.emit(EventType.ERROR, message=str(e))

    def _touch_thread(self, thread_id: str) -> None:
        """记录会话活跃时间，超上限时整体淘汰最久未活跃线程（LRU）

        MemorySaver 为支持时间旅行保留每个 superstep 的完整快照且无任何
        淘汰逻辑：长开服务随 会话数×轮数 无界增长，进程 RSS 只升不降
        （session_id 外部可控，压测/爬虫可放大）。当前线程刚被触碰必然是
        最新，淘汰永远不会误删在途会话。
        """
        self._thread_last_access[thread_id] = time.monotonic()
        if len(self._thread_last_access) <= config.checkpoint_max_threads:
            return

        oldest = min(self._thread_last_access, key=lambda tid: self._thread_last_access[tid])
        self._thread_last_access.pop(oldest, None)
        try:
            self.checkpointer.delete_thread(oldest)
            logger.info(f"检查点 LRU 淘汰最久未活跃线程: {oldest}")
        except Exception as e:  # noqa: BLE001 - 淘汰失败不影响主流程
            logger.debug(f"检查点淘汰失败（忽略）: {e}")

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

    @property
    def _store(self) -> SessionStore:
        """会话读写视图：每次从当前 checkpointer 派生（测试可整体替换）"""
        return SessionStore(self.checkpointer)

    def snapshot(self, session_id: str) -> dict:
        """读取会话消息历史（转换逻辑见 services.session_store）"""
        return {"messages": self._store.read_messages(session_id)}

    def reset(self, session_id: str) -> bool:
        """清空指定会话的检查点"""
        return self._store.clear(session_id)
