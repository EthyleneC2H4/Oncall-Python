"""X-API-Key 鉴权中间件测试：允许/拒绝矩阵 + 豁免路径 + 失败闭合"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import config
from app.middleware.auth import APIKeyMiddleware


@pytest.fixture
def client():
    """独立小应用（不挂业务路由），专注中间件行为"""
    app = FastAPI()

    @app.get("/api/ping")
    def ping():
        return {"pong": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/aiops/diagnose")
    def diagnose():
        return {"accepted": True}

    app.add_middleware(APIKeyMiddleware)
    return TestClient(app)


def _enable(monkeypatch, key: str = "secret-key"):
    monkeypatch.setattr(config, "auth_enabled", True)
    monkeypatch.setattr(config, "auth_api_key", key)


class TestDisabledByDefault:
    def test_all_paths_open_without_header(self, client, monkeypatch):
        monkeypatch.setattr(config, "auth_enabled", False)
        assert client.get("/api/ping").status_code == 200
        assert client.post("/api/aiops/diagnose").status_code == 200


class TestEnabled:
    def test_correct_key_passes(self, client, monkeypatch):
        _enable(monkeypatch)
        resp = client.get("/api/ping", headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 200
        assert resp.json() == {"pong": True}

    def test_wrong_key_rejected_401(self, client, monkeypatch):
        _enable(monkeypatch)
        resp = client.get("/api/ping", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["code"] == 401
        assert "X-API-Key" in body["detail"]

    def test_missing_header_rejected_401(self, client, monkeypatch):
        _enable(monkeypatch)
        assert client.get("/api/ping").status_code == 401
        assert client.post("/api/aiops/diagnose").status_code == 401


class TestExemptPaths:
    @pytest.mark.parametrize("path", ["/health", "/docs", "/redoc", "/openapi.json"])
    def test_exempt_prefixes_stay_open_even_when_enabled(self, client, monkeypatch, path):
        _enable(monkeypatch)
        assert client.get(path).status_code in (200, 404)  # 404=路由不存在但已过鉴权层

    def test_exempt_is_prefix_match(self, client, monkeypatch):
        """/healthz 这类前缀命中同样豁免；/api/vitals 不豁免"""
        _enable(monkeypatch)
        assert client.get("/healthz").status_code == 404  # 过了鉴权才谈得上 404
        assert client.get("/api/vitals").status_code == 401


class TestFailClosed:
    def test_empty_configured_key_rejects_everything(self, client, monkeypatch):
        """配置缺失（auth_api_key 为空）时宁可全拒，不放行空密钥匹配"""
        _enable(monkeypatch, key="")
        assert client.get("/api/ping").status_code == 401


class TestPreflightPassthrough:
    def test_options_preflight_passes_without_key_when_enabled(self, client, monkeypatch):
        """浏览器 CORS 预检不带自定义头，鉴权开启时必须放行 OPTIONS 否则前端全挂"""
        _enable(monkeypatch)

        resp = client.options(
            "/api/ping",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code != 401

    def test_non_preflight_options_still_requires_key(self, client, monkeypatch):
        """普通 OPTIONS（无 access-control-request-method 头）不享受豁免"""
        _enable(monkeypatch)
        assert client.options("/api/ping").status_code == 401
