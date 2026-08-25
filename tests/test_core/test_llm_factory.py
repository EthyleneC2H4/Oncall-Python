"""LLMFactory 降级链的 429 限流语义测试

免费档运营常态：429 不是故障。回归点：
- 有限次退避重试后成功 → 正常返回，不计熔断失败；
- 重试耗尽 → RateLimitExhausted，跳过同源备用模型（共享配额必同样 429），
  不计熔断失败，直接落缓存/模板；
- 非 429 异常语义不变 → 记熔断失败、尝试备用。
"""

import pytest

from app.config import config
from app.core.circuit_breaker import BREAKER_LLM, get_breaker
from app.core.degradation import DegradationLevel
from app.core.llm_factory import LLMFactory


class FakeRateLimit(Exception):
    """带 status_code=429 的假限流异常（_is_rate_limit 的 getattr 分支）"""

    def __init__(self):
        super().__init__("429 Too Many Requests")
        self.status_code = 429


class FakeResult:
    content = "ok"
    usage_metadata = {"input_tokens": 1, "output_tokens": 2}


class FakeLLM:
    """按脚本逐次抛出/返回；脚本耗尽后重复最后一项"""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, prompt: str):
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_models(monkeypatch, primary: FakeLLM, backup: FakeLLM) -> None:
    """主/备模型注入：按 model 参数路由到对应 FakeLLM"""

    def fake_create(cls, **kwargs):
        return primary if kwargs.get("model") == config.rag_model else backup

    monkeypatch.setattr(LLMFactory, "create_chat_model", classmethod(fake_create))


@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    breaker = get_breaker(BREAKER_LLM)
    breaker.reset()
    yield
    breaker.reset()


@pytest.fixture()
def _zero_backoff(monkeypatch):
    """退避间隔归零：测语义不测等待时长（仅降级链测试需要）"""
    monkeypatch.setattr(LLMFactory, "_rate_limit_delay", classmethod(lambda cls, e, a: 0.0))


class TestRateLimitRetryThenSuccess:
    @pytest.mark.asyncio
    async def test_retry_then_success_returns_normal(self, monkeypatch, _zero_backoff):
        primary, backup = FakeLLM([FakeRateLimit(), FakeResult()]), FakeLLM([])
        _patch_models(monkeypatch, primary, backup)

        content, level = await LLMFactory.invoke_with_fallback("唯一提示词-重试成功")

        assert (content, level) == ("ok", DegradationLevel.NONE)
        assert primary.calls == 2  # 首撞 429 → 退避后重试成功
        assert backup.calls == 0
        assert get_breaker(BREAKER_LLM).failure_count == 0  # 限流不计熔断失败


class TestRateLimitExhausted:
    @pytest.mark.asyncio
    async def test_exhausted_skips_backup_and_breaker(self, monkeypatch, _zero_backoff):
        primary, backup = FakeLLM([FakeRateLimit()]), FakeLLM([FakeResult()])
        _patch_models(monkeypatch, primary, backup)

        content, level = await LLMFactory.invoke_with_fallback("唯一提示词-耗尽", use_cache=False)

        assert level is DegradationLevel.TEMPLATE  # 无缓存可用，直落模板
        assert primary.calls == 2  # 初次 + 一次退避重试（_RATE_LIMIT_RETRIES=1）
        assert backup.calls == 0  # 同源免费档备用模型被跳过
        assert get_breaker(BREAKER_LLM).failure_count == 0  # 限流不计熔断失败


class TestNonRateLimitSemanticsUnchanged:
    @pytest.mark.asyncio
    async def test_generic_error_records_failure_and_tries_backup(
        self, monkeypatch, _zero_backoff
    ):
        primary = FakeLLM([RuntimeError("连接被拒")])
        backup = FakeLLM([FakeResult()])
        _patch_models(monkeypatch, primary, backup)

        content, level = await LLMFactory.invoke_with_fallback("普通故障提示词")

        assert (content, level) == ("ok", DegradationLevel.BACKUP_MODEL)
        assert backup.calls == 1
        assert get_breaker(BREAKER_LLM).failure_count == 1  # 真故障照常计失败


class TestRateLimitDelay:
    def test_honors_retry_after_capped(self):
        exc = Exception("x")
        exc.retry_after = 999  # 分钟级窗口等不起，封顶生效
        assert LLMFactory._rate_limit_delay(exc, 0) == LLMFactory._RATE_LIMIT_BACKOFF_CAP

        exc.retry_after = 3
        assert LLMFactory._rate_limit_delay(exc, 0) == 3.0

    def test_exponential_backoff_within_cap(self):
        exc = Exception("x")
        delay = LLMFactory._rate_limit_delay(exc, 10)
        assert 0 <= delay <= LLMFactory._RATE_LIMIT_BACKOFF_CAP
