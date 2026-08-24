"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""

import time
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.mcp_client import get_mcp_tools
from app.agent.runtime.toolsets import local_toolkit
from app.config import config
from app.core.llm_factory import LLMFactory
from app.models.plan import StructuredPlan, looks_like_plan, parse_plan
from app.services.context_assembler import context_assembler
from app.services.knowledge_graph_service import knowledge_graph_service
from app.services.query_router import query_router
from app.tools import (
    retrieve_with_hyde,
    retrieve_with_rewrite_and_rerank,
)

from .state import PlanExecuteState
from .utils import format_tools_description


class Plan(BaseModel):
    """计划的回退输出格式（扁平字符串列表——结构化输出失败时的双保险）"""

    steps: list[str] = Field(
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


async def _plan_structured(llm, chain_input: dict[str, Any]) -> StructuredPlan:
    """结构化规划：优先 StructuredPlan 输出；失败回退扁平 List[str] 再容错包装

    双保险依据：嵌套 JSON 的结构化输出对模型稳定性要求高，
    扁平字符串列表几乎总能成功，parse_plan 保证旧形态永远可表示。
    """
    try:
        structured_chain = planner_prompt | llm.with_structured_output(StructuredPlan)
        result = await structured_chain.ainvoke(chain_input)
        if not looks_like_plan(result):
            # provider 返回异常对象等非计划形态：视为结构化失败走回退，
            # 而非被容错解析当成一行文本步骤（那会掩盖上游故障）
            raise TypeError(f"意外的结构化输出类型: {type(result).__name__}")
        plan = result if isinstance(result, StructuredPlan) else parse_plan(result)
        if not plan.steps:
            raise ValueError("结构化输出为空计划")
        return plan
    except Exception as structured_err:
        logger.warning(f"结构化规划失败（{structured_err}），回退扁平列表模式")

    fallback_chain = planner_prompt | llm.with_structured_output(Plan)
    legacy = await fallback_chain.ainvoke(chain_input)
    steps = legacy.steps if isinstance(legacy, Plan) else legacy.get("steps", [])
    return parse_plan(steps)


async def planner(state: PlanExecuteState) -> dict[str, Any]:
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
                    context_str, _ = await retrieve_with_hyde(
                        input_text, top_k=strategy["rag_top_k"]
                    )
                else:
                    # 快速增强检索：Rewrite + BM25 + Rerank（无 HyDE，适合诊断类）
                    logger.info("使用快速增强检索 (Rewrite+BM25+Rerank)")
                    context_str, _ = await retrieve_with_rewrite_and_rerank(
                        input_text, top_k=strategy["rag_top_k"]
                    )
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
                all_keywords = keywords + [
                    "CPU",
                    "内存",
                    "memory",
                    "磁盘",
                    "disk",
                    "响应",
                    "slow",
                    "不可用",
                    "unavailable",
                    "OOM",
                ]
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
        # 获取本地工具（清单单一事实源见 runtime.toolsets）
        local_tools = local_toolkit()

        # 获取 MCP 工具（短 TTL 缓存，一次运行内复用）
        mcp_tools = await get_mcp_tools()

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

        # 步骤3.5: 召回历史事故的语义/程序记忆（经验复用闭环；失败安全）
        try:
            from app.core.context_engine import format_memory_block
            from app.services.memory import MemoryType, memory_service

            if memory_service.enabled:
                recalled = await memory_service.recall(
                    input_text, types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
                )
                memory_block = format_memory_block(recalled)
                if memory_block:
                    experience_context = (
                        experience_context
                        + "\n\n## 历史记忆（过往相似事件沉淀的经验）\n"
                        + memory_block
                    ).strip()
                    logger.info(f"Planner 注入历史记忆 {len(recalled)} 条")
        except Exception as e:
            logger.warning(f"记忆召回失败（忽略）: {e}")

        # 步骤4: 创建 LLM 并生成结构化计划（P3）
        llm = LLMFactory.create_chat_model(
            model=config.rag_model,
            temperature=0,
            streaming=False,
        )

        chain_input = {
            "messages": [("user", input_text)],
            "tools_description": tools_description,
            "experience_context": experience_context,
        }

        structured = await _plan_structured(llm, chain_input)
        plan_steps = structured.legacy_strings

        logger.info(f"计划已生成（{structured.source_format}），共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(structured.steps, 1):
            tool_tag = f" [tool={step.tool}]" if step.tool else ""
            logger.info(f"  步骤{i}{tool_tag}: {step.description}")

        # 构建诊断事件
        events = []
        events.append(
            {
                "timestamp": time.time(),
                "event_type": "routing",
                "agent": "planner",
                "action": f"查询意图分类: {intent}",
                "result_summary": f"关键词: {', '.join(keywords)}",
                "duration_ms": 0,
            }
        )
        if kg_context:
            events.append(
                {
                    "timestamp": time.time(),
                    "event_type": "kg_query",
                    "agent": "planner",
                    "action": "知识图谱查询",
                    "result_summary": kg_context[:200],
                    "duration_ms": 0,
                }
            )
        if experience_docs:
            events.append(
                {
                    "timestamp": time.time(),
                    "event_type": "rag_retrieve",
                    "agent": "planner",
                    "action": f"文档检索 (HyDE={strategy['use_hyde']})",
                    "result_summary": f"检索到 {len(experience_docs)} 字符的相关文档",
                    "duration_ms": 0,
                }
            )

        return {
            "plan": plan_steps,
            "plan_structured": [step.model_dump() for step in structured.steps],
            "kg_context": kg_context,
            "query_intent": intent,
            "diagnosis_events": events,
        }

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        # 返回一个默认计划
        return {
            "plan": ["收集相关信息", "分析数据", "生成报告"],
            # 兜底计划为纯文本步骤：显式清空结构化视图，维持逐位置对齐不变量
            "plan_structured": [],
            "diagnosis_events": [
                {
                    "timestamp": time.time(),
                    "event_type": "reasoning",
                    "agent": "planner",
                    "action": "生成计划失败，使用默认计划",
                    "result_summary": str(e),
                    "duration_ms": 0,
                }
            ],
        }
