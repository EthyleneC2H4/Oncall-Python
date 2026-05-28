"""请求限流中间件

基于滑动窗口的内存限流器。

限流规则：
- 全局：100 req/min
- 单 session：10 req/min
- AIOps 诊断：5 req/min（计算密集）
"""

import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from loguru import logger


class SlidingWindowCounter:
    """滑动窗口计数器"""

    def __init__(self, window_seconds: int = 60, max_requests: int = 100):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._requests: list[float] = []

    def is_allowed(self) -> bool:
        """检查是否允许请求"""
        now = time.time()
        cutoff = now - self.window_seconds

        # 清理过期记录
        self._requests = [t for t in self._requests if t > cutoff]

        if len(self._requests) >= self.max_requests:
            return False

        self._requests.append(now)
        return True

    @property
    def current_count(self) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        return sum(1 for t in self._requests if t > cutoff)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """请求限流中间件"""

    # 限流规则
    GLOBAL_LIMIT = 100        # 全局: 100 req/min
    SESSION_LIMIT = 10        # 单 session: 10 req/min
    AIOPS_LIMIT = 5           # AIOps: 5 req/min

    # AIOps 路径标识
    AIOPS_PATHS = {"/api/aiops", "/api/multi-diagnose"}

    def __init__(self, app):
        super().__init__(app)
        self._global_counter = SlidingWindowCounter(60, self.GLOBAL_LIMIT)
        self._session_counters: dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(60, self.SESSION_LIMIT)
        )
        self._aiops_counter = SlidingWindowCounter(60, self.AIOPS_LIMIT)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # 只对 API 路径限流
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        # 健康检查和静态资源不限流
        if path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        # 全局限流
        if not self._global_counter.is_allowed():
            logger.warning(f"全局限流触发: {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "code": 429,
                    "message": "请求过于频繁，请稍后重试",
                    "data": {"retry_after_seconds": 60},
                },
            )

        # AIOps 路径限流
        if path in self.AIOPS_PATHS:
            if not self._aiops_counter.is_allowed():
                logger.warning(f"AIOps 限流触发: {path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": 429,
                        "message": "诊断请求过于频繁，请稍后重试",
                        "data": {"retry_after_seconds": 60},
                    },
                )

        # Session 限流（从请求体或 query 中提取 session_id）
        session_id = request.query_params.get("session_id", "global")
        if not self._session_counters[session_id].is_allowed():
            logger.warning(f"Session 限流触发: session={session_id}, path={path}")
            return JSONResponse(
                status_code=429,
                content={
                    "code": 429,
                    "message": "当前会话请求过于频繁，请稍后重试",
                    "data": {"retry_after_seconds": 60},
                },
            )

        return await call_next(request)
