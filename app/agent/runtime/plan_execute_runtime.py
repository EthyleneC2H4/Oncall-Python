"""Plan-Execute-Replan 运行时 — 包装 AIOps StateGraph，产出统一结构化事件流

从 aiops_service.py 提取图构建逻辑；流式执行改为「deadline 内逐事件 yield」，
修复旧实现的假流式缺陷（wait_for 内先消费完整流再逐个 yield，
导致前端在整个工作流结束前收不到任何事件）。

节点状态增量 → 事件的映射规则（与旧 SSE 契约一一对应，见
app/api/event_translator.py 的 golden 快照）：
- planner   → PLAN_CREATED（计划 + 可选 kg_context/query_intent/diagnosis_events）
- executor  → STEP_END（past_steps 非空）或 STEP_START（开始执行）
- replanner → REPORT（已生成最终响应）或 REPLAN（继续/调整计划）
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agent.aiops import PlanExecuteState, executor, planner, replanner
from app.agent.runtime.base import AgentRuntime, default_registry
from app.agent.runtime.events import AgentEvent, AgentEventEmitter, EventType
from app.config import config
from app.services.session_store import SessionStore

# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"

# 步骤结果预览截断长度
_STEP_PREVIEW_CHARS = 300


def translate_graph_update(
    update: dict[str, Any],
    emitter: AgentEventEmitter,
    *,
    steps_done_base: int = 0,
) -> list[AgentEvent]:
    """把单个节点的状态增量翻译为结构化事件（纯函数，便于测试）

    Args:
        update: astream(stream_mode="updates") 的单个 chunk，{node_name: node_output}
        emitter: 当前 run 的事件发射器
        steps_done_base: 本 chunk 之前已累计完成的步骤数（STEP_END 进度用；
            executor 增量只含本次步骤，须由调用方累加）

    Returns:
        list[AgentEvent]: 0 或多个事件
    """
    events: list[AgentEvent] = []
    for node_name, node_output in update.items():
        if node_name == NODE_PLANNER:
            events.extend(_translate_planner(node_output, emitter))
        elif node_name == NODE_EXECUTOR:
            events.extend(
                _translate_executor(node_output, emitter, steps_done_base=steps_done_base)
            )
        elif node_name == NODE_REPLANNER:
            events.extend(_translate_replanner(node_output, emitter))
    return events


def _translate_planner(state: dict | None, emitter: AgentEventEmitter) -> list[AgentEvent]:
    """planner 输出 → PLAN_CREATED"""
    if not state:
        return [emitter.emit(EventType.STEP_START, stage="planner", message="规划节点执行中")]

    payload: dict[str, Any] = {
        "plan": state.get("plan", []),
    }
    # P3：结构化计划透传（事件层只增不改）
    if state.get("plan_structured"):
        payload["plan_structured"] = state["plan_structured"]
    # 与旧 _format_planner_event 一致：仅在有内容时附带上下文字段
    if state.get("kg_context"):
        payload["kg_context"] = state["kg_context"]
    if state.get("query_intent"):
        payload["query_intent"] = state["query_intent"]
    if state.get("diagnosis_events"):
        payload["diagnosis_events"] = state["diagnosis_events"]

    return [
        emitter.emit(
            EventType.PLAN_CREATED,
            message=f"执行计划已制定，共 {len(state.get('plan', []))} 个步骤",
            **payload,
        )
    ]


def _translate_executor(
    state: dict | None,
    emitter: AgentEventEmitter,
    *,
    steps_done_base: int = 0,
) -> list[AgentEvent]:
    """executor 输出 → STEP_END（有已完成步骤）或 STEP_START（开始执行）"""
    if not state:
        return [emitter.emit(EventType.STEP_START, stage="executor", message="执行节点运行中")]

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    if past_steps:
        last_step, last_result = past_steps[-1]
        result_text = str(last_result)
        return [
            emitter.emit(
                EventType.STEP_END,
                current_step=last_step,
                result_preview=result_text[:_STEP_PREVIEW_CHARS],
                # 节点增量只含本次步骤；进度须叠加此前累计
                steps_done=steps_done_base + len(past_steps),
                remaining_steps=len(plan),
            )
        ]
    return [emitter.emit(EventType.STEP_START, stage="executor", message="开始执行步骤")]


def _translate_replanner(state: dict | None, emitter: AgentEventEmitter) -> list[AgentEvent]:
    """replanner 输出 → REPORT（最终响应）或 REPLAN（继续/调整计划）"""
    if not state:
        return [emitter.emit(EventType.STEP_START, stage="replanner", message="评估节点运行中")]

    response = state.get("response", "")
    plan = state.get("plan", [])

    if response:
        return [emitter.emit(EventType.REPORT, report=response)]
    return [
        emitter.emit(
            EventType.REPLAN,
            message=f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
            remaining_steps=len(plan),
        )
    ]


class PlanExecuteRuntime(AgentRuntime):
    """Plan-Execute-Replan 运行时"""

    name = "plan_execute"

    def __init__(self, checkpointer: MemorySaver | None = None):
        """初始化服务"""
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()
        default_registry.register(self)
        logger.info("PlanExecuteRuntime 初始化完成")

    @property
    def _store(self) -> SessionStore:
        """会话读写视图：每次从当前 checkpointer 派生（测试可整体替换）"""
        return SessionStore(self.checkpointer)

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)  # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划

        workflow.set_entry_point(NODE_PLANNER)

        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)  # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)  # executor -> replanner

        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER, should_continue, {NODE_EXECUTOR: NODE_EXECUTOR, END: END}
        )

        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def run(self, task: str, session_id: str = "default") -> AsyncIterator[AgentEvent]:
        """在整体 deadline 内逐事件流式执行工作流

        与旧实现的本质区别：astream 每产出一个节点增量就立刻 yield，
        而不是等整个工作流跑完再补发。
        """
        emitter = AgentEventEmitter(session_id=session_id)

        logger.info(f"[会话 {session_id}] 开始执行任务: {task}")

        try:
            # 跨运行残留清理：past_steps/diagnosis_events/error_context 是
            # operator.add 追加通道，检查点若残留上一任务的增量，会被
            # initial_state 的空列表「追加」而非覆盖，导致同 session 第二个
            # 任务带着旧历史跑（误触发强制 respond/循环检测）。每次 run 前
            # 清掉该会话线程，任务级状态只属于本次运行。
            self._store.clear_best_effort(session_id)

            initial_state: PlanExecuteState = {
                "input": task,
                "plan": [],
                "plan_structured": [],
                "past_steps": [],
                "response": "",
                "kg_context": "",
                "query_intent": "",
                "diagnosis_events": [],
                "error_context": [],
                "degradation_level": "none",
                "session_id": session_id,  # 观测关联：工具痕迹按会话归组（P4）
            }

            config_dict = {"configurable": {"thread_id": session_id}}

            timed_out = False
            steps_done_total = 0  # 跨 chunk 累计的已完成步骤数（STEP_END 进度）
            try:
                async with asyncio.timeout(config.workflow_timeout_seconds):
                    async for update in self.graph.astream(
                        input=initial_state, config=config_dict, stream_mode="updates"
                    ):
                        for event in translate_graph_update(
                            update, emitter, steps_done_base=steps_done_total
                        ):
                            logger.debug(f"[会话 {session_id}] 事件 #{event.seq}: {event.type}")
                            yield event
                        steps_done_total += len(
                            (update.get(NODE_EXECUTOR) or {}).get("past_steps", []) or []
                        )
            except TimeoutError:
                timed_out = True
                logger.warning(
                    f"[会话 {session_id}] 工作流超时 ({config.workflow_timeout_seconds}s)，"
                    "生成部分报告"
                )

            # 读取最终状态（超时场景下为已完成的部分进度）
            final_state = self.graph.get_state(config_dict)
            response = ""
            if final_state and final_state.values:
                response = final_state.values.get("response", "")

            if timed_out and not response:
                response = (
                    "# 诊断超时\n\n"
                    f"诊断流程超过 {config.workflow_timeout_seconds} 秒限制，已自动终止。\n"
                    "请稍后重试或简化诊断任务。"
                )

            yield emitter.emit(
                EventType.COMPLETE,
                message="任务执行完成" if not timed_out else "任务超时，已生成部分报告",
                response=response,
                timed_out=timed_out,
            )

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield emitter.emit(EventType.ERROR, message=f"任务执行出错: {str(e)}")

    def snapshot(self, session_id: str) -> dict:
        """读取工作流最终状态快照"""
        try:
            from langchain_core.runnables import RunnableConfig

            run_config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = self.graph.get_state(run_config)
            if final_state and final_state.values:
                return {"values": dict(final_state.values)}
            return {"values": {}}
        except Exception as e:
            logger.error(f"读取工作流快照失败: {session_id}, 错误: {e}")
            return {"values": {}}

    def reset(self, session_id: str) -> bool:
        """清空指定会话的工作流检查点"""
        return self._store.clear(session_id)
