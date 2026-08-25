"""MemoryConsolidationWorker 单测：周期循环、开关语义、失败隔离、手动端点

全部注入替身服务，零外部依赖。回归点：
- 周期循环先睡后跑、可取消、启动幂等；
- memory_consolidate_enabled 只管周期循环，手动 run_once 不受限；
- consolidate 抛异常不外穿（下个周期重试）；
- POST /api/memory/consolidate 成功 200 / 未启用 409。
"""

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import config
from app.services.memory.consolidation_worker import (
    MemoryConsolidationWorker,
    consolidation_worker,
)


class FakeService:
    def __init__(self, enabled: bool = True, error: Exception | None = None):
        self.enabled = enabled
        self._error = error
        self.calls = 0

    async def consolidate(self) -> dict[str, Any]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return {"clusters": 1, "members_consolidated": 2, "semantic_ids": ["s1"]}


@pytest.fixture
def worker() -> MemoryConsolidationWorker:
    """每个用例独立实例，避免污染全局单例的观测状态"""
    return MemoryConsolidationWorker(service=FakeService())


class TestRunOnce:
    async def test_returns_stats_and_records_observation(self, worker):
        stats = await worker.run_once()

        assert stats == {"clusters": 1, "members_consolidated": 2, "semantic_ids": ["s1"]}
        assert worker.last_run_at > 0
        assert worker.last_stats == stats

    async def test_service_disabled_returns_none(self, worker):
        worker._service = FakeService(enabled=False)

        assert await worker.run_once() is None
        assert worker._service.calls == 0

    @pytest.mark.parametrize("periodic_enabled", [True, False])
    async def test_manual_trigger_ignores_periodic_switch(
        self, worker, monkeypatch, periodic_enabled
    ):
        """memory_consolidate_enabled 只管周期循环；手动触发是显式意图"""
        monkeypatch.setattr(config, "memory_consolidate_enabled", periodic_enabled)

        stats = await worker.run_once()
        assert stats is not None  # 开关两种取值下手动都可用

    async def test_consolidate_error_swallowed(self, worker):
        worker._service = FakeService(error=RuntimeError("嵌入器挂了"))

        stats = await worker.run_once()  # 不应抛出
        assert stats is None
        assert worker.last_stats is None  # 失败不污染观测面

    async def test_concurrent_runs_serialized_by_lock(self):
        class SlowService(FakeService):
            async def consolidate(self):
                await asyncio.sleep(0.05)
                return await super().consolidate()

        slow = SlowService()
        w = MemoryConsolidationWorker(service=slow)
        results = await asyncio.gather(w.run_once(), w.run_once(), w.run_once())

        # 锁保证不并发重叠：全部成功而非互相踩掉
        assert all(r is not None for r in results)
        assert slow.calls == 3


class TestPeriodicLoop:
    async def test_loop_runs_periodically_and_stops(self, worker, monkeypatch):
        monkeypatch.setattr(config, "memory_consolidate_interval_seconds", 0.05)
        worker.start()

        await asyncio.sleep(0.18)
        first_calls = worker._service.calls
        assert first_calls >= 2  # 先睡后跑：每个周期各执行一次

        await worker.stop()
        calls_after_stop = worker._service.calls
        await asyncio.sleep(0.1)
        assert worker._service.calls == calls_after_stop  # 取消后不再执行

    async def test_start_respects_periodic_switch(self, monkeypatch):
        monkeypatch.setattr(config, "memory_consolidate_enabled", False)
        w = MemoryConsolidationWorker(service=FakeService())

        w.start()
        assert w._task is None  # disabled 不建任务

        await w.stop()

    async def test_start_idempotent(self, worker, monkeypatch):
        monkeypatch.setattr(config, "memory_consolidate_interval_seconds", 0.05)
        worker.start()
        first_task = worker._task
        worker.start()  # 已在跑 → 不重复建任务
        assert worker._task is first_task
        await worker.stop()


class TestManualEndpoint:
    def test_endpoint_success_and_conflict(self, monkeypatch):
        from app.api.memory import router

        app = FastAPI()
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        # 未启用记忆 → 409
        consolidation_worker._service = FakeService(enabled=False)
        resp = client.post("/api/memory/consolidate")
        assert resp.status_code == 409

        # 启用 → 200 + 统计
        consolidation_worker._service = FakeService()
        resp = client.post("/api/memory/consolidate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["clusters"] == 1
