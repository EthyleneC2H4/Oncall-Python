"""Circuit Breaker 熔断器单元测试

测试覆盖：状态转换、熔断触发、半开恢复、冷却时间。
"""

import time

import pytest

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_breaker,
)


class TestCircuitBreakerStateTransitions:
    """状态转换测试"""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        assert cb.state == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        cb.before_call()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failures_open_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            cb.before_call()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_failures_below_threshold_stays_closed(self):
        cb = CircuitBreaker("test", failure_threshold=5, cooldown_seconds=60)
        for _ in range(4):
            cb.before_call()
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_open_to_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.before_call()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.before_call()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.before_call()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.before_call()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.before_call()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRejection:
    """熔断拒绝测试"""

    def test_before_call_raises_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=60)
        cb.before_call()
        cb.record_failure()
        with pytest.raises(CircuitOpenError, match="Circuit breaker for 'test' is OPEN"):
            cb.before_call()

    def test_before_call_allows_when_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.before_call()
        cb.record_failure()
        time.sleep(0.02)
        # 不应抛异常
        cb.before_call()

    def test_reset_restores_closed(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=60)
        cb.before_call()
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerRegistry:
    """注册中心测试"""

    def test_get_breaker_creates_once(self):
        b1 = get_breaker("my_test")
        b2 = get_breaker("my_test")
        assert b1 is b2

    def test_get_breaker_different_names(self):
        b1 = get_breaker("test_a")
        b2 = get_breaker("test_b")
        assert b1 is not b2

    def test_pre_registered_breakers(self):
        # 与 circuit_breaker.py 底部预注册清单保持一致
        for name in ["llm", "embedding", "rerank", "milvus", "mcp_cls", "mcp_monitor"]:
            cb = get_breaker(name)
            assert cb.name == name
            assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerConcurrency:
    """线程安全测试"""

    def test_concurrent_success_counting(self):
        import threading

        cb = CircuitBreaker("concurrent_test", failure_threshold=5, cooldown_seconds=60)
        results = []

        def call():
            try:
                cb.before_call()
                cb.record_success()
                results.append(True)
            except CircuitOpenError:
                results.append(False)

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert cb.state == CircuitState.CLOSED
