"""MCP 工具列表 TTL 缓存测试（P1.1：每次 run 拉取一次而非每步）"""

import pytest

from app.agent.mcp_client import (
    get_mcp_tools,
    reset_mcp_tools_cache,
)


@pytest.fixture(autouse=True)
def _clean_tools_cache():
    reset_mcp_tools_cache()
    yield
    reset_mcp_tools_cache()


@pytest.mark.asyncio
async def test_cached_within_ttl(monkeypatch):
    """TTL 内重复调用只拉取一次"""
    fetch_count = 0

    class FakeClient:
        async def get_tools(self):
            nonlocal fetch_count
            fetch_count += 1
            return [f"tool_{fetch_count}"]

    async def fake_client(*args, **kwargs):
        return FakeClient()

    monkeypatch.setattr("app.agent.mcp_client.get_mcp_client_with_retry", fake_client)

    first = await get_mcp_tools()
    second = await get_mcp_tools()

    assert fetch_count == 1
    assert first is second


@pytest.mark.asyncio
async def test_refresh_bypasses_cache(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.n = 0

        async def get_tools(self):
            self.n += 1
            return [f"tool_{self.n}"]

    client = FakeClient()

    async def fake_client(*args, **kwargs):
        return client

    monkeypatch.setattr("app.agent.mcp_client.get_mcp_client_with_retry", fake_client)

    await get_mcp_tools()
    refreshed = await get_mcp_tools(refresh=True)

    assert refreshed == ["tool_2"]


@pytest.mark.asyncio
async def test_expired_ttl_refetches(monkeypatch):
    calls = []

    class FakeClient:
        async def get_tools(self):
            calls.append(1)
            return ["t"]

    async def fake_client(*args, **kwargs):
        return FakeClient()

    monkeypatch.setattr("app.agent.mcp_client.get_mcp_client_with_retry", fake_client)

    await get_mcp_tools(ttl_seconds=0.01)
    import asyncio

    await asyncio.sleep(0.02)
    await get_mcp_tools(ttl_seconds=0.01)

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_fetch_failure_falls_back_to_stale_cache(monkeypatch):
    """拉取失败时回退过期缓存而不是抛异常"""

    class FlakyClient:
        def __init__(self, fail: bool = False):
            self.fail = fail

        async def get_tools(self):
            if self.fail:
                raise RuntimeError("MCP server down")
            return ["log_query"]

    client = FlakyClient()

    async def fake_client(*args, **kwargs):
        return client

    monkeypatch.setattr("app.agent.mcp_client.get_mcp_client_with_retry", fake_client)

    await get_mcp_tools()  # 预热缓存

    client.fail = True  # 过期 + 拉取失败
    stale = await get_mcp_tools(ttl_seconds=0.0)

    assert stale == ["log_query"]


@pytest.mark.asyncio
async def test_fetch_failure_without_cache_raises(monkeypatch):
    async def fake_client(*args, **kwargs):
        raise RuntimeError("MCP server down")

    monkeypatch.setattr("app.agent.mcp_client.get_mcp_client_with_retry", fake_client)

    with pytest.raises(RuntimeError, match="MCP server down"):
        await get_mcp_tools()
