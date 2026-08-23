"""降级策略模块单元测试

测试覆盖：降级等级枚举、严重度映射、降级信息构建、模板响应。
与 app/core/degradation.py 的真实 API 对齐：
- severity_of(level) 接收单个等级（非列表）
- DegradationInfo 字段: is_degraded/level/active_degradations/unavailable_services/user_message
"""


from app.core.degradation import (
    DEGRADATION_MESSAGES,
    TEMPLATE_RESPONSES,
    DegradationInfo,
    DegradationLevel,
    build_degradation_info,
    severity_of,
)


class TestDegradationLevel:
    """降级等级枚举测试"""

    def test_all_levels_have_messages(self):
        # NONE 表示无降级，不需要用户提示；BACKUP_MODEL 对用户透明，提示为 None
        for level in DegradationLevel:
            if level == DegradationLevel.NONE:
                continue
            assert level in DEGRADATION_MESSAGES
            message = DEGRADATION_MESSAGES[level]
            assert message is None or message  # None 或非空文案

    def test_level_count(self):
        """确保 10 级降级"""
        assert len(DegradationLevel) == 10

    def test_none_level_is_no_degradation(self):
        assert severity_of(DegradationLevel.NONE) == "none"


class TestSeverity:
    """严重度测试"""

    def test_no_degradation_none(self):
        assert severity_of(DegradationLevel.NONE) == "none"

    def test_minor_degradation(self):
        # 缓存命中 / 备用模型 / 跳过重排 对用户影响轻微
        assert severity_of(DegradationLevel.CACHED) == "minor"
        assert severity_of(DegradationLevel.BACKUP_MODEL) == "minor"
        assert severity_of(DegradationLevel.NO_RERANK) == "minor"

    def test_moderate_degradation(self):
        assert severity_of(DegradationLevel.BM25_ONLY) == "moderate"
        assert severity_of(DegradationLevel.BM25_RAW) == "moderate"
        assert severity_of(DegradationLevel.SINGLE_AGENT) == "moderate"

    def test_severe_degradation(self):
        assert severity_of(DegradationLevel.TEMPLATE) == "severe"
        assert severity_of(DegradationLevel.DIRECT_RAG) == "severe"
        assert severity_of(DegradationLevel.KG_ONLY) == "severe"


class TestBuildDegradationInfo:
    """降级信息构建测试"""

    def test_empty_levels(self):
        info = build_degradation_info([])
        assert isinstance(info, DegradationInfo)
        assert info.is_degraded is False
        assert info.level == "none"
        assert info.active_degradations == []
        assert info.user_message is None

    def test_none_only_levels(self):
        info = build_degradation_info([DegradationLevel.NONE])
        assert info.is_degraded is False

    def test_single_level(self):
        info = build_degradation_info([DegradationLevel.BACKUP_MODEL])
        assert info.is_degraded is True
        assert "backup_model" in info.active_degradations
        assert info.level == "minor"

    def test_user_message_takes_first_available(self):
        info = build_degradation_info(
            [
                DegradationLevel.BACKUP_MODEL,  # 提示为 None（对用户透明）
                DegradationLevel.NO_RERANK,  # 有用户提示
            ]
        )
        assert info.user_message == DEGRADATION_MESSAGES[DegradationLevel.NO_RERANK]

    def test_unavailable_services_recorded(self):
        info = build_degradation_info([DegradationLevel.BM25_ONLY], unavailable=["milvus"])
        assert info.unavailable_services == ["milvus"]


class TestTemplateResponses:
    """模板响应测试"""

    def test_all_intents_have_template(self):
        for intent in ["DIAGNOSTIC", "KNOWLEDGE", "CHITCHAT", "COMPLEX"]:
            assert intent in TEMPLATE_RESPONSES
            assert TEMPLATE_RESPONSES[intent]

    def test_diagnostic_template_mentions_root_cause(self):
        assert "根因" in TEMPLATE_RESPONSES["DIAGNOSTIC"]

    def test_chitchat_template_is_friendly(self):
        assert "运维" in TEMPLATE_RESPONSES["CHITCHAT"]
