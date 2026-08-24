"""降级等级定义与模板响应

统一的降级等级枚举、用户提示文案、模板响应。
所有降级路径共享此模块以保持一致性。
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class DegradationLevel(StrEnum):
    """降级等级"""

    NONE = "none"  # 正常
    BACKUP_MODEL = "backup_model"  # 使用备用 LLM
    NO_RERANK = "no_rerank"  # 跳过 Rerank
    BM25_ONLY = "bm25_only"  # 仅 BM25 关键词检索
    BM25_RAW = "bm25_raw"  # BM25 + 原始查询（无改写）
    KG_ONLY = "kg_only"  # 仅知识图谱
    CACHED = "cached"  # 缓存结果
    SINGLE_AGENT = "single_agent"  # 多 Agent 降级为单 Agent
    DIRECT_RAG = "direct_rag"  # 跳过 Agent，直接 RAG
    TEMPLATE = "template"  # 静态模板响应


class DegradationInfo(BaseModel):
    """附加在 API 响应中的降级状态"""

    is_degraded: bool = False
    level: str = "none"  # none / minor / moderate / severe
    active_degradations: list[str] = Field(default_factory=list)
    unavailable_services: list[str] = Field(default_factory=list)
    user_message: str | None = None


# ---------- 用户提示映射 ----------

DEGRADATION_MESSAGES: dict[str, str | None] = {
    DegradationLevel.BACKUP_MODEL: None,  # 对用户透明
    DegradationLevel.NO_RERANK: "结果排序精度可能略有不同。",
    DegradationLevel.BM25_ONLY: "语义检索暂时不可用，当前基于关键词匹配。",
    DegradationLevel.BM25_RAW: "语义检索暂时不可用，当前基于关键词匹配。",
    DegradationLevel.KG_ONLY: "文档检索暂时不可用，当前基于知识图谱分析。",
    DegradationLevel.CACHED: "显示的是近期相同查询的缓存结果。",
    DegradationLevel.SINGLE_AGENT: "正在使用简化诊断模式。",
    DegradationLevel.DIRECT_RAG: "自动诊断暂时不可用，展示相关参考文档。",
    DegradationLevel.TEMPLATE: "当前无法执行完整分析，请稍后重试。",
}

# ---------- 静态模板响应 ----------

TEMPLATE_RESPONSES: dict[str, str] = {
    "DIAGNOSTIC": (
        "当前无法执行完整诊断分析。基于您的告警描述，建议检查：\n"
        "1. 近期变更记录\n2. 资源使用率趋势\n3. 应用日志中的错误信息\n"
        "请稍后重试以获取根因定位的完整诊断报告。"
    ),
    "KNOWLEDGE": "知识库检索暂时不可用，请稍后重试或直接查阅运维文档。",
    "CHITCHAT": "运维助手当前遇到一些连接问题，请稍后再试。",
    "COMPLEX": "复杂分析功能暂时不可用，请尝试简化查询或稍后重试。",
}


def get_template_response(intent: str) -> str:
    return TEMPLATE_RESPONSES.get(intent, TEMPLATE_RESPONSES["KNOWLEDGE"])


def severity_of(level: DegradationLevel) -> str:
    """降级等级 → 严重程度"""
    if level in (DegradationLevel.NONE,):
        return "none"
    if level in (
        DegradationLevel.BACKUP_MODEL,
        DegradationLevel.NO_RERANK,
        DegradationLevel.CACHED,
    ):
        # 备用模型对用户透明、缓存结果内容一致，均属轻度降级
        return "minor"
    if level in (
        DegradationLevel.BM25_ONLY,
        DegradationLevel.BM25_RAW,
        DegradationLevel.SINGLE_AGENT,
    ):
        return "moderate"
    return "severe"


def build_degradation_info(
    levels: list[DegradationLevel],
    unavailable: list[str] | None = None,
) -> DegradationInfo:
    """从降级等级列表构建 DegradationInfo"""
    if not levels or levels == [DegradationLevel.NONE]:
        return DegradationInfo()

    worst = max(levels, key=lambda lvl: list(DegradationLevel).index(lvl))
    messages = [DEGRADATION_MESSAGES.get(lvl) for lvl in levels]
    user_msg = next((m for m in messages if m), None)

    return DegradationInfo(
        is_degraded=True,
        level=severity_of(worst),
        active_degradations=[lvl.value for lvl in levels if lvl != DegradationLevel.NONE],
        unavailable_services=unavailable or [],
        user_message=user_msg,
    )
