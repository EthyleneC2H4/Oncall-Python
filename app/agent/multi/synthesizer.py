"""Synthesizer — 汇总多 Agent 发现，交叉验证，生成结构化报告"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config


def _get_synthesis_prompt() -> str:
    """获取 Synthesizer Prompt，优先从版本化管理器加载"""
    try:
        from app.core.prompt_manager import prompt_manager
        template = prompt_manager.get("synthesizer")
        if template:
            return template.content
    except Exception:
        pass

    return """你是一个高级运维诊断综合分析师。你需要汇总多个专业 Agent 的分析结果，交叉验证各方发现，生成最终诊断报告。

## 原始告警
{alert_input}

## 各 Agent 分析结果

### 日志分析 Agent（置信度: {log_confidence}）
{log_findings}

### 指标检查 Agent（置信度: {metric_confidence}）
{metric_findings}

### 知识检索 Agent（置信度: {knowledge_confidence}）
{knowledge_findings}

## 你的任务

1. **交叉验证**：检查多个 Agent 的发现是否一致，标注一致性高的结论和有矛盾的地方
2. **根因定位**：综合所有证据，给出最可能的根因（标注置信度）
3. **处置建议**：按紧急/短期/长期分类给出处置建议
4. **级联风险**：基于知识图谱的级联预测，提醒需要关注的后续风险

请输出结构化的 Markdown 诊断报告。
"""


SYNTHESIS_PROMPT = _get_synthesis_prompt()


class Synthesizer:
    """汇总器：融合多 Agent 发现，交叉验证，生成报告"""

    def __init__(self):
        self.llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0,
        )

    async def synthesize(self, alert_input: str, findings: list) -> str:
        """汇总多 Agent 发现，生成结构化报告

        Args:
            alert_input: 原始告警描述
            findings: AgentFinding 列表

        Returns:
            str: Markdown 格式的综合诊断报告
        """
        # 提取各 Agent 的发现
        agent_data = {
            "log_findings": "（无结果）",
            "log_confidence": 0.0,
            "metric_findings": "（无结果）",
            "metric_confidence": 0.0,
            "knowledge_findings": "（无结果）",
            "knowledge_confidence": 0.0,
        }

        for finding in findings:
            if finding.error:
                result_text = f"（执行失败: {finding.error}）"
            else:
                result_text = finding.findings or "（无结果）"

            if finding.agent_name == "log_analyst":
                agent_data["log_findings"] = result_text
                agent_data["log_confidence"] = finding.confidence
            elif finding.agent_name == "metric_inspector":
                agent_data["metric_findings"] = result_text
                agent_data["metric_confidence"] = finding.confidence
            elif finding.agent_name == "knowledge_retriever":
                agent_data["knowledge_findings"] = result_text
                agent_data["knowledge_confidence"] = finding.confidence

        try:
            # 格式化置信度为百分比字符串
            formatted_data = dict(agent_data)
            for key in ("log_confidence", "metric_confidence", "knowledge_confidence"):
                formatted_data[key] = f"{formatted_data[key]:.0%}"

            prompt = SYNTHESIS_PROMPT.format(
                alert_input=alert_input,
                **formatted_data,
            )

            messages = [
                SystemMessage(content="你是高级运维诊断综合分析师，善于交叉验证和结构化分析。"),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            logger.info("Synthesizer 报告生成完成")
            return response.content

        except Exception as e:
            logger.error(f"Synthesizer 汇总失败: {e}")
            # 降级：简单拼接各 Agent 结果
            return self._fallback_report(alert_input, findings)

    def _fallback_report(self, alert_input: str, findings: list) -> str:
        """降级报告：简单拼接"""
        parts = [f"# 诊断报告（降级模式）\n\n## 告警\n{alert_input}\n"]

        for finding in findings:
            status = "✅" if not finding.error else "❌"
            parts.append(f"## {status} {finding.agent_name}")
            if finding.error:
                parts.append(f"执行失败: {finding.error}")
            else:
                parts.append(finding.findings or "无结果")
            parts.append(f"*置信度: {finding.confidence:.0%}, 耗时: {finding.duration_ms:.0f}ms*\n")

        return "\n\n".join(parts)
