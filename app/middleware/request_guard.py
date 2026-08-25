"""请求守卫中间件

功能：
- 请求标准化（统一 request_id、session_id）
- 幂等控制（60s 内相同 method+path+request_id 返回缓存）
- 输入安全检测集成
- 请求审计日志
"""

import hashlib
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.cache import TTLCache
from app.middleware.input_guard import input_guard

# 幂等缓存：(method, path, request_id) → response，60s TTL
_idempotency_cache = TTLCache(name="idempotency", maxsize=1000, ttl_seconds=60)


def _idempotency_key(method: str, path: str, request_id: str) -> str:
    """幂等键绑定 method+path：request_id 由客户端任填，裸用作键会让
    不同请求（乃至不同用户）在 60s 内互相拿到对方响应"""
    return hashlib.sha256(f"{method}:{path}:{request_id}".encode()).hexdigest()


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
            idem_key = _idempotency_key(request.method, path, request_id)
            cached = _idempotency_cache.get(idem_key)
            if isinstance(cached, Response):
                logger.info(f"幂等命中: request_id={request_id}")
                return cached
        else:
            idem_key = ""

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

        # 缓存响应用于幂等控制（仅成功的非流式 POST：SSE 的 body_iterator
        # 只能被消费一次，缓存重放会得到空流或两个请求共享迭代器交叉穿插）。
        # 另注意：BaseHTTPMiddleware 拿到的响应体本身也是一次性流，直接缓存
        # 原对象的话回放时 body 已被首次请求耗尽、得到空体 200——必须先物化
        # 字节并重建 Response 再缓存。
        is_streaming = "text/event-stream" in response.headers.get("content-type", "")
        body_iterator = getattr(response, "body_iterator", None)  # 流式子类才携带
        if request.method == "POST" and response.status_code == 200 and not is_streaming:
            if body_iterator is None:  # 纯字节响应无需物化，可直接缓存
                _idempotency_cache.set(idem_key, response)
            else:
                body = b"".join([chunk async for chunk in body_iterator])
                response = Response(
                    content=body,
                    status_code=response.status_code,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() != "content-length"
                    },
                )
                _idempotency_cache.set(idem_key, response)

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(int(latency_ms))

        return response
