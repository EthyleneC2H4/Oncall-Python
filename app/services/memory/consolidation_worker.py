"""定时记忆巩固 worker

周期性执行 memory_service.consolidate()（情景 → 语义贪心聚类合并），把重复
出现的操作经验沉淀为可召回的语义记忆——经验复用闭环的定时侧。

设计约束：
- 开关与周期走配置：memory_consolidate_enabled 只管周期循环；
  手动触发（POST /api/memory/consolidate）是显式用户意图，不受该开关限制。
- 失败不致命：巩固是读-算-写批处理，异常记日志后等下个周期，绝不影响
  在线记忆读写路径。
- 周期与手动共用 run_once()，asyncio.Lock 防两者重叠（candidates→mark
  非原子，并发巩固可能重复合并同一簇）。
"""

import asyncio
import time
from typing import Any

from loguru import logger

from app.config import config


class MemoryConsolidationWorker:
    """情景→语义记忆巩固的周期执行器"""

    def __init__(self, service: Any | None = None):
        self._service = service  # 测试注入；None 时惰性解析全局单例
        self._task: asyncio.Task[None] | None = None
        self._run_lock = asyncio.Lock()
        # 观测面：最近一次巩固的时间与统计（/health 或调试用）
        self.last_run_at: float = 0.0
        self.last_stats: dict[str, Any] | None = None

    def _resolve_service(self) -> Any:
        if self._service is not None:
            return self._service
        from app.services.memory.service import memory_service

        return memory_service

    async def run_once(self) -> dict[str, Any] | None:
        """手动/周期共用的单次巩固入口

        Returns:
            consolidate 统计 dict；记忆服务未就绪或本次失败返回 None
            （调用方据此区分 409 与成功）。
        """
        svc = self._resolve_service()
        if not getattr(svc, "enabled", False):
            return None
        async with self._run_lock:  # 周期与手动互斥，防重复合并同一簇
            try:
                stats: dict[str, Any] = await svc.consolidate()
            except Exception as e:  # noqa: BLE001 - 巩固失败不影响在线路径
                logger.warning(f"记忆巩固失败（下个周期重试）: {e}")
                return None
            self.last_run_at = time.time()
            self.last_stats = stats
            if stats.get("clusters"):
                logger.info(f"记忆巩固完成: {stats}")
            return stats

    def start(self) -> None:
        """启动周期循环（disabled 时不建任务；重复调用幂等）"""
        if not config.memory_consolidate_enabled:
            logger.info("记忆巩固 worker 未启用（memory_consolidate_enabled=false）")
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="memory-consolidation")
        logger.info(f"记忆巩固 worker 已启动（间隔 {config.memory_consolidate_interval_seconds}s）")

    async def stop(self) -> None:
        """取消周期任务并等待退出"""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(config.memory_consolidate_interval_seconds)
            # 先睡后跑：启动瞬间没有需要巩固的增量，也不拖慢服务就绪
            await self.run_once()


# 全局巩固 worker 单例（lifespan 中 start/stop）
consolidation_worker = MemoryConsolidationWorker()
