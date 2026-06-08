"""降级策略模块单元测试

测试覆盖：降级等级枚举、严重度映射、降级信息构建。
"""

import pytest

from app.core.degradation import (
    DegradationLevel,
    DegradationInfo,
    build_degradation_info,
    severity_of,
    DEGRADATION_MESSAGES,
    TEMPLATE_RESPONSES,
)


class TestDegradationLevel:
    """降级等级枚举测试"""

    def test_all_levels_have_messages(self):
        for level in DegradationLevel:
            assert level in DEGRADATION_MESSAGES
            assert DEGRADATION_MESSAGES[level]  # 非空

    def test_level_count(self):
        """确保 10 级降级"""
        assert len(DegradationLevel) == 10

    def test_none_level_is_no_degradation(self):
        assert severity_of([DegradationLevel.NONE]) == "none"


class TestSeverity:
    """严重度测试"""

    def test_no_degradation_none(self):
        assert severity_of([]) == "none"
        assert severity_of([DegradationLevel.NONE]) == "none"

    def test_minor_degradation(self):
        assert severity_of([DegradationLevel.CACHED]) == "minor"

    def test_moderate_degradation(self):
        assert severity_of([DegradationLevel.BACKUP_MODEL]) == "moderate"
        assert severity_of([DegradationLevel.BM25_ONLY]) == "moderate"

    def test_severe_degradation(self):
        assert severity_of([DegradationLevel.TEMPLATE]) == "severe"
        assert severity_of([DegradationLevel.DIRECT_RAG]) == "severe"

    def test_worst_severity_wins(self):
        assert severity_of([
            DegradationLevel.NO_RERANK,
            DegradationLevel.TEMPLATE,
        ]) == "severe"


class TestBuildDegradationInfo:
    """降级信息构建测试"""

    def test_empty_levels(self):
        info = build_degradation_info([])
        assert isinstance(info, DegradationInfo)
        assert info.degradation_levels == []
        assert info.severity == "none"
        assert info.description == ""

    def test_single_level(self):
        info = build_degradation_info([DegradationLevel.BACKUP_MODEL])
        assert DegradationLevel.BACKUP_MODEL in info.degradation_levels
        assert info.severity == "moderate"

    def test_description_concatenation(self):
        info = build_degradation_info([
            DegradationLevel.NO_RERANK,
            DegradationLevel.BM25_ONLY,
        ])
        assert "Rerank" in info.description or "rerank" in info.description.lower()
        assert "BM25" in info.description

    def test_fallback_enabled(self):
        info = build_degradation_info([DegradationLevel.TEMPLATE])
        assert info.fallback_enabled is True


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
