"""统一服务健康注册中心

集中追踪所有外部依赖的实时健康状态，支持：
- 周期性后台探针 (asyncio task)
- 服务状态三级：HEALTHY / DEGRADED / DOWN
- 与 CircuitBreaker 联动：DOWN 时熔断器直接 OPEN
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Callable, Awaitable, Optional

from loguru import logger


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ServiceHealth:
    """单个服务的健康信息"""

    def __init__(self, name: str):
        self.name = name
        self.status = ServiceStatus.HEALTHY
        self.last_check: float = 0.0
        self.last_success: Optional[float] = None
        self.consecutive_failures: int = 0
        self.latency_ms: Optional[float] = None

    def record_success(self, latency_ms: float) -> None:
        self.status = ServiceStatus.HEALTHY
        self.last_check = time.time()
        self.last_success = self.last_check
        self.consecutive_failures = 0
        self.latency_ms = latency_ms

    def record_failure(self) -> None:
        self.last_check = time.time()
        self.consecutive_failures += 1
        if self.consecutive_failures >= 5:
            self.status = ServiceStatus.DOWN
        elif self.consecutive_failures >= 2:
            self.status = ServiceStatus.DEGRADED

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
        }


# 探针函数类型：返回 True 表示健康
ProbeFunc = Callable[[], Awaitable[bool]]


class HealthRegistry:
    """全局健康注册中心"""

    def __init__(self):
        self._services: Dict[str, ServiceHealth] = {}
        self._probes: Dict[str, ProbeFunc] = {}
        self._probe_task: Optional[asyncio.Task] = None
        self._interval_seconds: float = 30.0

    def register(self, name: str, probe: Optional[ProbeFunc] = None) -> None:
        """注册服务（可选探针函数）"""
        if name not in self._services:
            self._services[name] = ServiceHealth(name)
        if probe:
            self._probes[name] = probe

    def is_available(self, name: str) -> bool:
        """服务是否可用（HEALTHY 或 DEGRADED）"""
        svc = self._services.get(name)
        if svc is None:
            return True  # 未注册的服务默认可用
        return svc.status != ServiceStatus.DOWN

    def get_status(self, name: str) -> ServiceStatus:
        svc = self._services.get(name)
        return svc.status if svc else ServiceStatus.HEALTHY

    def mark_success(self, name: str, latency_ms: float = 0.0) -> None:
        svc = self._services.get(name)
        if svc:
            svc.record_success(latency_ms)

    def mark_failure(self, name: str) -> None:
        svc = self._services.get(name)
        if svc:
            svc.record_failure()

    def mark_down(self, name: str) -> None:
        """直接标记为 DOWN（如启动时连接失败）"""
        svc = self._services.get(name)
        if svc:
            svc.status = ServiceStatus.DOWN
            svc.consecutive_failures = 5

    def get_all_status(self) -> Dict[str, dict]:
        return {name: svc.to_dict() for name, svc in self._services.items()}

    # ---------- 后台探针 ----------

    async def start_probes(self, interval: float = 30.0) -> None:
        """启动后台周期性探针"""
        self._interval_seconds = interval
        if self._probe_task is None or self._probe_task.done():
            self._probe_task = asyncio.create_task(self._probe_loop())
            logger.info(f"健康探针已启动，间隔 {interval}s")

    async def stop_probes(self) -> None:
        if self._probe_task and not self._probe_task.done():
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
            logger.info("健康探针已停止")

    async def _probe_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval_seconds)
                await self._run_all_probes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康探针异常: {e}")

    async def _run_all_probes(self) -> None:
        for name, probe in self._probes.items():
            try:
                start = time.time()
                healthy = await asyncio.wait_for(probe(), timeout=5.0)
                latency = (time.time() - start) * 1000
                if healthy:
                    self.mark_success(name, latency)
                else:
                    self.mark_failure(name)
            except asyncio.TimeoutError:
                self.mark_failure(name)
                logger.warning(f"健康探针超时: {name}")
            except Exception as e:
                self.mark_failure(name)
                logger.warning(f"健康探针失败: {name} - {e}")


# 全局单例
health_registry = HealthRegistry()
