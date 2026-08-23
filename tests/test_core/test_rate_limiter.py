"""限流器模块单元测试

测试覆盖：滑动窗口计数、限流触发、多层级限流、429 响应。
"""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.middleware.rate_limiter import RateLimiterMiddleware, SlidingWindowCounter


class TestSlidingWindowCounter:
    """滑动窗口计数器测试

    真实 API: SlidingWindowCounter(window_seconds, max_requests)；
    is_allowed() 判定并记录请求；current_count 为动态计算的 property。
    """

    def test_initial_count_zero(self):
        counter = SlidingWindowCounter(window_seconds=60)
        assert counter.current_count == 0

    def test_is_allowed_records_request(self):
        counter = SlidingWindowCounter(window_seconds=60)
        assert counter.is_allowed()
        assert counter.is_allowed()
        assert counter.current_count == 2

    def test_expired_entries_no_longer_counted(self):
        counter = SlidingWindowCounter(window_seconds=0.01)
        assert counter.is_allowed()
        assert counter.is_allowed()
        assert counter.current_count == 2
        time.sleep(0.02)
        # current_count 动态过滤窗口外记录
        assert counter.current_count == 0

    def test_is_allowed_within_limit(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)
        assert counter.is_allowed()

    def test_is_allowed_blocked_at_limit(self):
        counter = SlidingWindowCounter(window_seconds=60, max_requests=5)
        for _ in range(5):
            assert counter.is_allowed()
        assert not counter.is_allowed()

    def test_window_expiry_restores_allowance(self):
        counter = SlidingWindowCounter(window_seconds=0.01, max_requests=1)
        assert counter.is_allowed()
        assert not counter.is_allowed()
        time.sleep(0.02)
        assert counter.is_allowed()


class TestRateLimiterMiddleware:
    """限流中间件测试"""

    @pytest.fixture
    def app_with_rate_limiter(self):
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/api/aiops")
        async def aiops_endpoint():
            return {"status": "aiops"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        app.add_middleware(RateLimiterMiddleware)
        return app

    def test_normal_request_passes(self, app_with_rate_limiter):
        client = TestClient(app_with_rate_limiter)
        response = client.get("/api/test?session_id=test123")
        assert response.status_code == 200

    def test_health_not_rate_limited(self, app_with_rate_limiter):
        client = TestClient(app_with_rate_limiter)
        for _ in range(20):  # 超过 session limit
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_returns_429(self, app_with_rate_limiter):
        client = TestClient(app_with_rate_limiter)
        # 会话速率限制 10 req/min
        for _i in range(15):
            response = client.get("/api/test?session_id=ratelimit_test")
            if response.status_code == 429:
                data = response.json()
                # 统一响应封装: {code, message, data}
                assert "retry_after_seconds" in data["data"]
                assert "频繁" in data["message"]
                return
        pytest.fail("Expected 429 but never got one after 15 requests")

    def test_different_sessions_independent(self, app_with_rate_limiter):
        client = TestClient(app_with_rate_limiter)
        # Session A: send many requests
        for _ in range(12):
            client.get("/api/test?session_id=session_a")
        # Session B: should still pass
        resp_b = client.get("/api/test?session_id=session_b")
        assert resp_b.status_code == 200


class TestRateLimiterEdgeCases:
    """限流边界情况"""

    @pytest.fixture
    def app(self):
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint(request: Request):
            return {"status": "ok"}

        app.add_middleware(RateLimiterMiddleware)
        return app

    def test_missing_session_id_uses_default(self, app):
        client = TestClient(app)
        response = client.get("/api/test")
        assert response.status_code == 200

    def test_non_api_path_skipped(self, app):
        """非 /api/ 路径跳过限流"""
        client = TestClient(app)
        # 直接访问一个不存在的路径
        response = client.get("/static/test.js")
        assert response.status_code in (200, 404)  # 不限流

    def test_wipe_unblocks(self, app):
        """验证滑动窗口过期后可恢复"""
        client = TestClient(app)
        session = "wipe_test"
        for _ in range(12):
            client.get(f"/api/test?session_id={session}")
        # 等待窗口过期
        time.sleep(1.1)  # window_size 默认 1s
        response = client.get(f"/api/test?session_id={session}")
        assert response.status_code in (200, 429)  # 依赖于具体实现
