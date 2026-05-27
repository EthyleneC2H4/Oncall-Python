"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现，增加单步超时控制
"""

import asyncio
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge, query_alert_graph, predict_alert_cascade
from app.agent.mcp_client import get_mcp_client_with_retry
from app.harness.agent_rules import AGENT_RULES
from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    async def _execute_step():
        """内部执行逻辑，被超时包装"""
        # 获取本地工具（含知识图谱工具）
        local_tools = [
            get_current_time,
            retrieve_knowledge,
            query_alert_graph,
            predict_alert_cascade,
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        all_tools = local_tools + mcp_tools

        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
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
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            return final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            logger.info("LLM 未调用工具，直接返回结果")
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

    # 带超时的步骤执行
    step_timeout = config.step_timeout_seconds
    try:
        result = await asyncio.wait_for(_execute_step(), timeout=step_timeout)
        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        return {
            "plan": plan[1:],
            "past_steps": [(task, result)],
        }

    except asyncio.TimeoutError:
        logger.warning(f"步骤执行超时 ({step_timeout}s): {task}")
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"步骤执行超时（{step_timeout}s）")],
            "error_context": [{"step": task, "error_type": "timeout", "error_msg": f"超时 {step_timeout}s"}],
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
            "error_context": [{"step": task, "error_type": "exception", "error_msg": str(e)}],
        }
