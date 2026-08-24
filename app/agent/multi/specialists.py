"""专业 Agent - 日志分析、指标检查、知识检索

三个专业 Agent 并行执行各自的分析任务，
通过 MCP 工具或本地工具获取数据。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.config import config
from app.core.llm_factory import LLMFactory
from app.tools import predict_alert_cascade, query_alert_graph, retrieve_knowledge
from app.tools.filters import tools_for_role


class BaseSpecialistAgent:
    """专业 Agent 基类"""

    name: str = "base"
    system_prompt: str = ""
    confidence: float = 0.5

    def __init__(self):
        self.llm = LLMFactory.create_chat_model(
            streaming=False,
            model=config.rag_model,
            temperature=0,
        )

    async def analyze(self, alert_input: str) -> str:
        """执行分析任务（子类实现）"""
        raise NotImplementedError


class LogAnalystAgent(BaseSpecialistAgent):
    """日志分析 Agent — 通过 MCP:CLS 查询日志，提取异常模式"""

    name = "log_analyst"
    system_prompt = """你是一个专业的日志分析专家。
你的任务是通过日志查询工具，分析当前系统的异常日志，找出关键错误模式。

分析流程：
1. 使用日志查询工具搜索与告警相关的错误日志
2. 提取关键错误信息、异常堆栈、时间模式
3. 总结日志中发现的异常模式

输出格式：
- 关键发现（3-5 条最重要的日志线索）
- 错误模式（是否有重复错误、错误频率趋势）
- 时间关联（错误与告警时间的关联）
"""

    async def analyze(self, alert_input: str) -> str:
        try:
            # 获取 MCP 工具（短 TTL 缓存），按角色过滤（单一事实源见 tools/filters.py）
            mcp_tools = await get_mcp_tools()
            log_tools = tools_for_role(mcp_tools, self.name)

            if log_tools:
                llm_with_tools = self.llm.bind_tools(log_tools)
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"请分析以下告警相关的系统日志：{alert_input}"),
                ]
                response = await llm_with_tools.ainvoke(messages)

                # 如果有工具调用，执行后汇总
                if hasattr(response, "tool_calls") and response.tool_calls:
                    from langgraph.prebuilt import ToolNode

                    tool_node = ToolNode(log_tools)
                    messages.append(response)
                    tool_results = await tool_node.ainvoke({"messages": messages})
                    messages.extend(tool_results["messages"])
                    final = await self.llm.ainvoke(messages)
                    self.confidence = 0.7
                    return str(final.content)
                else:
                    self.confidence = 0.5
                    return str(response.content)
            else:
                # 无日志工具时，基于 LLM 知识分析
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"（无日志查询工具可用）请基于经验分析以下告警可能的日志表现：{alert_input}"
                    ),
                ]
                response = await self.llm.ainvoke(messages)
                self.confidence = 0.3
                return str(response.content)

        except Exception as e:
            logger.error(f"LogAnalystAgent 分析失败: {e}")
            raise


class MetricInspectorAgent(BaseSpecialistAgent):
    """指标检查 Agent — 通过 MCP:Monitor 查询监控指标"""

    name = "metric_inspector"
    system_prompt = """你是一个专业的监控指标分析专家。
你的任务是通过指标查询工具，检查系统的关键监控指标，发现异常趋势。

分析流程：
1. 使用指标查询工具获取 CPU、内存、磁盘、网络等关键指标
2. 识别异常峰值、趋势变化、阈值突破
3. 关联多个指标之间的因果关系

输出格式：
- 异常指标（哪些指标超过阈值或趋势异常）
- 时间线（指标变化的关键时间点）
- 关联分析（指标之间的因果推断）
"""

    async def analyze(self, alert_input: str) -> str:
        try:
            # 获取 MCP 工具（短 TTL 缓存），按角色过滤（单一事实源见 tools/filters.py）
            mcp_tools = await get_mcp_tools()
            metric_tools = tools_for_role(mcp_tools, self.name)

            if metric_tools:
                llm_with_tools = self.llm.bind_tools(metric_tools)
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"请检查以下告警相关的系统监控指标：{alert_input}"),
                ]
                response = await llm_with_tools.ainvoke(messages)

                if hasattr(response, "tool_calls") and response.tool_calls:
                    from langgraph.prebuilt import ToolNode

                    tool_node = ToolNode(metric_tools)
                    messages.append(response)
                    tool_results = await tool_node.ainvoke({"messages": messages})
                    messages.extend(tool_results["messages"])
                    final = await self.llm.ainvoke(messages)
                    self.confidence = 0.7
                    return str(final.content)
                else:
                    self.confidence = 0.5
                    return str(response.content)
            else:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"（无指标查询工具可用）请基于经验分析以下告警可能的指标表现：{alert_input}"
                    ),
                ]
                response = await self.llm.ainvoke(messages)
                self.confidence = 0.3
                return str(response.content)

        except Exception as e:
            logger.error(f"MetricInspectorAgent 分析失败: {e}")
            raise


class KnowledgeRetrieverAgent(BaseSpecialistAgent):
    """知识检索 Agent — 从 RAG + KG 获取诊断知识"""

    name = "knowledge_retriever"
    system_prompt = """你是一个运维知识检索专家。
你的任务是从知识图谱和文档知识库中获取与当前告警相关的诊断知识。

分析流程：
1. 使用 query_alert_graph 查询告警的根因、处置和级联关系
2. 使用 predict_alert_cascade 预测可能的级联风险
3. 使用 retrieve_knowledge 检索相关的操作文档
4. 综合结构化知识和文档内容

输出格式：
- 知识图谱分析（根因列表、级联风险）
- 文档知识（相关操作步骤、最佳实践）
- 推荐处置方案（按优先级排序）
"""

    async def analyze(self, alert_input: str) -> str:
        try:
            # 本地知识工具按角色过滤（与 MCP 工具同一事实源）
            all_local = [query_alert_graph, predict_alert_cascade, retrieve_knowledge]
            local_tools = tools_for_role(all_local, self.name)
            llm_with_tools = self.llm.bind_tools(local_tools)

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"请检索以下告警的相关知识：{alert_input}"),
            ]
            response = await llm_with_tools.ainvoke(messages)

            if hasattr(response, "tool_calls") and response.tool_calls:
                from langgraph.prebuilt import ToolNode

                tool_node = ToolNode(local_tools)
                messages.append(response)
                tool_results = await tool_node.ainvoke({"messages": messages})
                messages.extend(tool_results["messages"])
                final = await llm_with_tools.ainvoke(messages)

                # 如果有二次工具调用，继续执行
                if hasattr(final, "tool_calls") and final.tool_calls:
                    messages.append(final)
                    tool_results2 = await tool_node.ainvoke({"messages": messages})
                    messages.extend(tool_results2["messages"])
                    final = await self.llm.ainvoke(messages)

                self.confidence = 0.8
                return str(final.content)
            else:
                self.confidence = 0.4
                return str(response.content)

        except Exception as e:
            logger.error(f"KnowledgeRetrieverAgent 分析失败: {e}")
            raise
