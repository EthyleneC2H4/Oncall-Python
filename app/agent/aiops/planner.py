"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""

import time
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge, retrieve_with_hyde, retrieve_with_rewrite_and_rerank, query_alert_graph, predict_alert_cascade
from app.agent.mcp_client import get_mcp_client_with_retry
from app.services.knowledge_graph_service import knowledge_graph_service
from app.services.query_router import query_router
from app.services.context_assembler import context_assembler
from .state import PlanExecuteState
from .utils import format_tools_description


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )


def _get_planner_prompt_text() -> str:
    """获取 Planner Prompt，优先从版本化管理器加载"""
    try:
        from app.core.prompt_manager import prompt_manager
        template = prompt_manager.get("planner")
        if template:
            logger.debug(f"加载 Planner Prompt 模板: v{template.version}")
            return template.content
    except Exception:
        pass

    return dedent("""
        作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

        可用工具列表（用于制定计划时参考）：

        {tools_description}

        注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

        {experience_context}

        对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
        - 将任务分解为逻辑上独立的步骤
        - 每个步骤应该明确使用哪些工具(如果需要工具的话)来获取信息, 最好能同时提供工具执行所需要的参数
        - 步骤之间应该有清晰的依赖关系
        - 步骤描述要具体、可操作
        - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

        示例输入："分析当前系统的性能问题"
        示例输出（假设有对应工具）：
        步骤1: 使用 get_metrics 工具收集系统的 CPU 和内存使用情况
        步骤2: 使用 query_logs 工具检查最近的错误日志
        步骤3: 使用 query_database 工具分析慢查询日志
        步骤4: 综合以上信息生成性能分析报告
    """).strip()


# Planner 提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _get_planner_prompt_text()),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        # 步骤0: 查询意图分类与路由
        intent, keywords = await query_router.route(input_text)
        strategy = query_router.get_retrieval_strategy(intent)
        logger.info(f"查询路由: intent={intent}, keywords={keywords}, strategy={strategy}")

        # 步骤1: 根据路由策略检索文档（含 Query Rewrite + BM25 + Rerank）
        experience_docs = ""
        if strategy["use_rag"]:
            logger.info("查询内部文档，寻找相关经验...")
            try:
                if strategy["use_hyde"]:
                    # 全链路增强检索：Rewrite + HyDE + BM25 + Rerank
                    logger.info("使用全链路增强检索 (Rewrite+HyDE+BM25+Rerank)")
                    context_str, _ = await retrieve_with_hyde(input_text, top_k=strategy["rag_top_k"])
                else:
                    # 快速增强检索：Rewrite + BM25 + Rerank（无 HyDE，适合诊断类）
                    logger.info("使用快速增强检索 (Rewrite+BM25+Rerank)")
                    context_str, _ = await retrieve_with_rewrite_and_rerank(input_text, top_k=strategy["rag_top_k"])
                if context_str and context_str.strip():
                    experience_docs = context_str
                    logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
                else:
                    logger.info("未找到相关经验文档")
            except Exception as e:
                logger.warning(f"查询内部文档失败: {e}")

        # 步骤1.5: 查询知识图谱获取结构化关联信息
        kg_context = ""
        if strategy["use_kg"]:
            try:
                # 优先使用路由提取的关键词，再尝试默认关键词列表
                all_keywords = keywords + ["CPU", "内存", "memory", "磁盘", "disk", "响应", "slow", "不可用", "unavailable", "OOM"]
                for keyword in all_keywords:
                    if keyword.lower() in input_text.lower():
                        kg_analysis = knowledge_graph_service.format_analysis_context(keyword)
                        if kg_analysis:
                            kg_context = kg_analysis
                            logger.info(f"知识图谱命中关键词: {keyword}")
                            break
            except Exception as e:
                logger.warning(f"知识图谱查询失败: {e}")

        # 步骤2: 获取可用工具列表
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge,
            query_alert_graph,
            predict_alert_cascade,
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

        # 步骤3: 使用上下文组装器格式化经验上下文
        experience_context = context_assembler.format_experience_context(
            kg_context=kg_context,
            rag_context=experience_docs,
        )

        # 步骤4: 创建 LLM 并生成计划
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": tools_description,
            "experience_context": experience_context
        })

        # 提取步骤列表
        if isinstance(plan_result, Plan):
            plan_steps = plan_result.steps
        else:
            # 如果返回的是字典，提取 steps 字段
            plan_steps = plan_result.get("steps", [])  # type: ignore

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        # 构建诊断事件
        events = []
        events.append({
            "timestamp": time.time(),
            "event_type": "routing",
            "agent": "planner",
            "action": f"查询意图分类: {intent}",
            "result_summary": f"关键词: {', '.join(keywords)}",
            "duration_ms": 0,
        })
        if kg_context:
            events.append({
                "timestamp": time.time(),
                "event_type": "kg_query",
                "agent": "planner",
                "action": "知识图谱查询",
                "result_summary": kg_context[:200],
                "duration_ms": 0,
            })
        if experience_docs:
            events.append({
                "timestamp": time.time(),
                "event_type": "rag_retrieve",
                "agent": "planner",
                "action": f"文档检索 (HyDE={strategy['use_hyde']})",
                "result_summary": f"检索到 {len(experience_docs)} 字符的相关文档",
                "duration_ms": 0,
            })

        return {
            "plan": plan_steps,
            "kg_context": kg_context,
            "query_intent": intent,
            "diagnosis_events": events,
        }

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        # 返回一个默认计划
        return {
            "plan": [
                "收集相关信息",
                "分析数据",
                "生成报告"
            ],
            "diagnosis_events": [{
                "timestamp": time.time(),
                "event_type": "reasoning",
                "agent": "planner",
                "action": "生成计划失败，使用默认计划",
                "result_summary": str(e),
                "duration_ms": 0,
            }],
        }
