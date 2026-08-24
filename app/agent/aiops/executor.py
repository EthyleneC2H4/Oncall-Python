"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现，增加单步超时控制
"""

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.agent.runtime.toolsets import local_toolkit
from app.config import config
from app.core.llm_factory import LLMFactory
from app.harness.agent_rules import AGENT_RULES
from app.tools.filters import tools_executable_without_approval

from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    P3 双路径：
    - 结构化步骤绑定了已知工具 → guarded_call 直接调用（省一次 mini-ReAct，
      且经权限/参数/确认门三道关卡）
    - 未绑定或找不到工具 → 回退既有 LLM mini-ReAct 路径
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])
    structured = state.get("plan_structured") or []

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 消费端对齐防御：两列视图数量不一致时整体弃用结构化视图，
    # 退化为纯 legacy 路径（宁可少一次直连优化，不可错位执行）
    if structured and len(structured) != len(plan):
        logger.warning(
            f"plan({len(plan)}) 与 plan_structured({len(structured)}) 数量错位，"
            "本次弃用结构化视图"
        )
        structured = []

    # 取出第一个步骤（结构化与旧列表按位置对齐；错位时以旧列表为准）
    task = plan[0]
    step_dict = structured[0] if structured else None
    if step_dict and str(step_dict.get("description") or "").strip():
        task = str(step_dict["description"])
    logger.info(f"当前任务: {task}" + (f" [tool={step_dict.get('tool')}]" if step_dict and step_dict.get("tool") else ""))

    async def _execute_direct(tool_name: str, args: dict) -> str:
        """P3 直连路径：经 guard 执行绑定工具，不走 LLM 决策"""
        from app.tools.guard import guarded_call

        local_tools = local_toolkit()
        mcp_tools = await get_mcp_tools()
        matched = next(
            (t for t in [*local_tools, *mcp_tools] if getattr(t, "name", "") == tool_name),
            None,
        )
        if matched is None:
            raise ValueError(f"步骤绑定的工具 '{tool_name}' 不在可用工具池中")

        # 观测关联 ID 随痕迹落盘（缺省空串——BFCL 会话过滤依赖非空 session_id）
        result = await guarded_call(
            matched, args,
            request_id=str(state.get("request_id", "")),
            session_id=str(state.get("session_id", "")),
        )
        if result.needs_confirmation:
            return (
                f"[待人工确认] 该步骤触发高风险操作确认门（action_id={result.action_id}）。"
                f"原因：{result.error}。请在审批接口批准后继续。"
            )
        if not result.ok:
            raise RuntimeError(result.error or "工具执行失败")
        return f"[直接工具调用 {tool_name}]\n{result.value}"

    async def _execute_react():
        """回退路径：mini-ReAct 由 LLM 自主选工具"""
        # 获取本地工具（含知识图谱工具；清单单一事实源见 runtime.toolsets）
        local_tools = local_toolkit()

        # 获取 MCP 工具（短 TTL 缓存，一次运行内复用）
        mcp_tools = await get_mcp_tools()

        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 确认门完整性：高风险工具不得进入自主执行池（ToolNode 直连执行会
        # 绕过 guarded_call，必须先经待审动作人工批准）
        all_tools = tools_executable_without_approval([*local_tools, *mcp_tools])

        llm = LLMFactory.create_chat_model(
            model=config.rag_model,
            temperature=0,
            streaming=False,
        )
        llm_with_tools = llm.bind_tools(all_tools)

        tool_node = ToolNode(all_tools)

        messages = [
            SystemMessage(content=f"""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务

{AGENT_RULES}"""),
            HumanMessage(content=f"请执行以下任务: {task}"),
        ]

        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            return (
                final_response.content
                if hasattr(final_response, "content")
                else str(final_response)
            )
        else:
            logger.info("LLM 未调用工具，直接返回结果")
            return llm_response.content if hasattr(llm_response, "content") else str(llm_response)

    async def _execute_step() -> tuple[str, bool]:
        """返回 (结果文本, 是否走了直连路径)"""
        if step_dict and step_dict.get("tool"):
            try:
                result = await _execute_direct(step_dict["tool"], step_dict.get("args") or {})
                return result, True
            except Exception as direct_err:  # noqa: BLE001 - 直连失败降级 mini-ReAct
                logger.warning(f"直连工具调用失败（{direct_err}），降级 mini-ReAct 路径")
        return await _execute_react(), False

    # 带超时的步骤执行
    step_timeout = config.step_timeout_seconds
    try:
        result, direct_used = await asyncio.wait_for(_execute_step(), timeout=step_timeout)
        logger.info(f"步骤执行完成（direct={direct_used}），结果长度: {len(result)}")

        return {
            "plan": plan[1:],
            "plan_structured": structured[1:] if len(structured) > 1 else [],
            "past_steps": [(task, result)],
        }

    except TimeoutError:
        logger.warning(f"步骤执行超时 ({step_timeout}s): {task}")
        return {
            "plan": plan[1:],
            "plan_structured": structured[1:] if len(structured) > 1 else [],
            "past_steps": [(task, f"步骤执行超时（{step_timeout}s）")],
            "error_context": [
                {"step": task, "error_type": "timeout", "error_msg": f"超时 {step_timeout}s"}
            ],
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "plan_structured": structured[1:] if len(structured) > 1 else [],
            "past_steps": [(task, f"执行失败: {str(e)}")],
            "error_context": [{"step": task, "error_type": "exception", "error_msg": str(e)}],
        }
