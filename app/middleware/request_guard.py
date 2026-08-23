"""请求守卫中间件

功能：
- 请求标准化（统一 request_id、session_id）
- 幂等控制（60s 内相同 request_id 返回缓存）
- 输入安全检测集成
- 请求审计日志
"""

import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.cache import TTLCache
from app.middleware.input_guard import input_guard

# 幂等缓存：request_id → response，60s TTL
_idempotency_cache = TTLCache(name="idempotency", maxsize=1000, ttl_seconds=60)


class RequestGuardMiddleware(BaseHTTPMiddleware):
    """请求守卫中间件"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 只处理 API 请求
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        start_time = time.time()

        # 注入 request_id
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.start_time = start_time

        # 幂等控制：检查是否重复请求
        if request.method == "POST":
            cached = _idempotency_cache.get(request_id)
            if isinstance(cached, Response):
                logger.info(f"幂等命中: request_id={request_id}")
                return cached

        # 对 POST 请求进行输入安全检测
        if request.method == "POST" and path not in ("/api/upload", "/api/index_directory"):
            try:
                body = await request.body()
                body_text = body.decode("utf-8", errors="ignore")

                if body_text:
                    is_safe, message, _ = input_guard.validate(body_text)
                    if not is_safe:
                        logger.warning(
                            f"输入安全检测拦截: request_id={request_id}, reason={message}"
                        )
                        return JSONResponse(
                            status_code=400,
                            content={
                                "code": 400,
                                "message": f"输入内容不安全: {message}",
                                "data": None,
                            },
                        )
            except Exception as e:
                logger.debug(f"输入检测跳过: {e}")

        # 执行请求
        response = await call_next(request)

        # 记录审计日志
        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[Audit] {request.method} {path} "
            f"request_id={request_id} "
            f"status={response.status_code} "
            f"latency={latency_ms:.0f}ms"
        )

        # 缓存响应用于幂等控制（仅成功的 POST）
        if request.method == "POST" and response.status_code == 200:
            _idempotency_cache.set(request_id, response)

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(int(latency_ms))

        return response
