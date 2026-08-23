"""结构化诊断报告模型

定义诊断过程中的事件流和最终报告的数据结构，
支持推理轨迹追踪和前端实时展示。
"""


from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """诊断证据"""

    source: str = Field(description="证据来源: KG | RAG | MCP:CLS | MCP:Monitor")
    content: str = Field(description="证据内容")
    relevance: float = Field(default=1.0, description="相关性评分 0-1")


class Action(BaseModel):
    """处置建议"""

    priority: str = Field(description="优先级: urgent | short_term | long_term")
    description: str = Field(description="处置描述")
    expected_effect: str = Field(default="", description="预期效果")


class DiagnosisReport(BaseModel):
    """结构化诊断报告"""

    alert_type: str = Field(default="", description="告警类型")
    alert_time: str = Field(default="", description="告警时间")
    affected_service: str = Field(default="", description="影响范围")

    root_cause: str = Field(default="", description="根因分析")
    confidence: float = Field(default=0.0, description="诊断置信度 0-1")
    evidence: list[Evidence] = Field(default_factory=list, description="证据链")

    cascade_risk: list[str] = Field(default_factory=list, description="级联风险预测")
    actions: list[Action] = Field(default_factory=list, description="处置建议")

    diagnosis_duration: float = Field(default=0.0, description="诊断耗时(秒)")
    sources: list[str] = Field(default_factory=list, description="参考来源")

    def to_markdown(self) -> str:
        """将诊断报告格式化为 Markdown"""
        parts = [f"# 诊断报告: {self.alert_type}\n"]

        if self.alert_time or self.affected_service:
            parts.append("## 基本信息")
            if self.alert_time:
                parts.append(f"- **告警时间**: {self.alert_time}")
            if self.affected_service:
                parts.append(f"- **影响范围**: {self.affected_service}")
            parts.append("")

        if self.root_cause:
            parts.append("## 根因分析")
            parts.append(f"**结论**: {self.root_cause}")
            parts.append(f"**置信度**: {self.confidence:.0%}")
            parts.append("")

        if self.evidence:
            parts.append("## 证据链")
            for i, ev in enumerate(self.evidence, 1):
                parts.append(f"{i}. [{ev.source}] {ev.content}")
            parts.append("")

        if self.cascade_risk:
            parts.append("## 级联风险")
            for risk in self.cascade_risk:
                parts.append(f"- ⚠️ {risk}")
            parts.append("")

        if self.actions:
            urgency_icon = {"urgent": "🔴", "short_term": "🟡", "long_term": "🟢"}
            parts.append("## 处置建议")
            for action in self.actions:
                icon = urgency_icon.get(action.priority, "⚪")
                parts.append(f"- {icon} **{action.description}**")
                if action.expected_effect:
                    parts.append(f"  预期效果: {action.expected_effect}")
            parts.append("")

        if self.diagnosis_duration > 0:
            parts.append(f"---\n*诊断耗时: {self.diagnosis_duration:.1f}s*")

        return "\n".join(parts)


class DiagnosisEvent(BaseModel):
    """诊断过程中的每个关键事件（推理轨迹）"""

    timestamp: float = Field(description="事件时间戳")
    event_type: str = Field(
        description="事件类型: kg_query | rag_retrieve | mcp_call | reasoning | routing"
    )
    agent: str = Field(
        default="system", description="执行者: planner | executor | router | knowledge_retriever"
    )
    action: str = Field(description="具体动作描述")
    result_summary: str = Field(default="", description="结果摘要")
    confidence: float | None = Field(default=None, description="置信度")
    duration_ms: float = Field(default=0.0, description="耗时(毫秒)")


class FeedbackRecord(BaseModel):
    """用户反馈记录"""

    session_id: str = Field(description="会话ID")
    message_index: int = Field(description="消息索引")
    feedback_type: str = Field(description="反馈类型: positive | negative | comment")
    comment: str = Field(default="", description="用户补充说明")
    actual_root_cause: str = Field(default="", description="用户提供的实际根因")
