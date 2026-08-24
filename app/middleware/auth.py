"""X-API-Key 静态密钥鉴权中间件

auth_enabled=False（默认）时完全放行 —— 本地开发零负担；
开启后除豁免路径（健康检查/文档/静态资源）外一律要求
Header `X-API-Key` 与 config.auth_api_key 匹配（常数时间比较）。
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import config

# 无需鉴权的前缀：探活、文档与静态资源保持开放（供 k8s probe / 浏览器直接访问）
_EXEMPT_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/static",
)

_HEADER = "X-API-Key"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """静态密钥鉴权（作品集诚实档位：不做 IAM/JWT/RBAC）"""

    async def dispatch(self, request: Request, call_next):
        # CORS 预检必须透传：本中间件位于最外层（限流之前），若拦截 OPTIONS，
        # 浏览器拿不到 ACAO 头，开启鉴权后所有跨域客户端整体不可用
        if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
            return await call_next(request)

        if not config.auth_enabled or not self._required(request.url.path):
            return await call_next(request)

        provided = request.headers.get(_HEADER, "")
        if not config.auth_api_key or not hmac.compare_digest(
            provided.encode(), config.auth_api_key.encode()
        ):
            return JSONResponse(
                status_code=401,
                content={"code": 401, "detail": "缺少或无效的 X-API-Key"},
            )
        return await call_next(request)

    @staticmethod
    def _required(path: str) -> bool:
        return not path.startswith(_EXEMPT_PREFIXES)
