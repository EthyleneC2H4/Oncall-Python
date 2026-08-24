"""长期记忆 - 单 worker 异步写队列

sqlite 写入必须串行（避免 SQLITE_BUSY），且不应阻塞事件循环。
本队列提供：
- start(): 启动单个消费者 task，按 FIFO 顺序执行提交的协程工厂
- submit(fn, *args): 提交异步操作，返回 Future 供调用方等待（可选）
- drain(): 等待队列清空（测试同步点）
- stop(): 优雅停止（哨兵后排空；任何路径退出都 fail-fast 清算残留 Future）

未 start() 时 submit 直接内联执行 —— 测试可同步直通，无需事件循环编排。

停止不变量：stop() 返回后，所有已提交 Future 必然已 resolve
（要么执行完毕、要么带 RuntimeError 失败），调用方不会永久悬挂；
worker 收到哨兵后会继续排空哨兵之前入队的剩余任务。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")

OpFactory = Callable[[], Coroutine[Any, Any, Any]]


class WriteQueue:
    """单 worker 异步写队列"""

    def __init__(self, name: str = "memory-write"):
        self.name = name
        self._queue: asyncio.Queue[tuple[OpFactory, asyncio.Future[Any]] | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    # ──────────────── 生命周期 ────────────────

    def start(self) -> None:
        """启动消费者 task（幂等）"""
        if self._worker is not None and not self._worker.done():
            return
        self._worker = asyncio.create_task(self._run(), name=f"{self.name}-worker")
        logger.debug(f"WriteQueue[{self.name}] worker 已启动")

    async def stop(self) -> None:
        """优雅停止：通知退出 → 等 worker 排空 → fail-fast 清算残留 Future

        超时则强制取消 worker，随后统一把队列中未完成的 Future 标记失败，
        保证调用方 await submit() 返回的 Future 永不悬挂。
        """
        if self._worker is None:
            return
        worker = self._worker
        await self._queue.put(None)  # 哨兵：worker 排空剩余任务后退出
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=5.0)
            self._worker = None  # 确认结束后才清引用
        except TimeoutError:
            logger.warning(f"WriteQueue[{self.name}] worker 停止超时，强制取消")
            worker.cancel()
            with _suppress_cancel():
                await asyncio.shield(worker)  # 等真正退出再清理
            self._worker = None
        except asyncio.CancelledError:
            # 调用方（如 uvicorn lifespan）被强制取消：不吞取消、不清引用——
            # worker 可能仍在跑，保持 running=True 让后续 submit 继续走队列保序；
            # 哨兵已入队，worker 会自行收尾
            raise
        finally:
            # 兜底清算：覆盖超时取消、外部 cancel 等非常规退出路径下
            # 残留在队列中的条目（正常路径队列为空，此循环为 no-op）
            await self._fail_pending()

    async def _fail_pending(self) -> None:
        """把队列中所有未完成条目的 Future 标记为失败并出队"""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()
            if item is None:
                continue  # 陈旧哨兵直接丢弃，避免毒化下一次 start()
            _, future = item
            if not future.done():
                future.set_exception(RuntimeError(f"WriteQueue[{self.name}] 已停止，操作未执行"))

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    # ──────────────── 提交 ────────────────

    def submit(self, fn: OpFactory) -> asyncio.Future[Any]:
        """提交一个异步操作；返回 Future

        - 队列未启动 → 内联直通执行（测试友好）
        - 队列已启动 → FIFO 排队，异常写入 Future 而不杀死 worker
        """
        if not self.running:
            return asyncio.ensure_future(fn())

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((fn, future))
        return future

    async def drain(self) -> None:
        """等待队列中已提交的操作全部执行完毕"""
        await self._queue.join()

    # ──────────────── worker ────────────────

    async def _run(self) -> None:
        stopping = False
        while True:
            if stopping and self._queue.empty():
                return  # 哨兵已到且积压排空：正常收尾
            item = await self._queue.get()
            try:
                if item is None:  # 哨兵：进入停止流程，继续排空剩余任务
                    stopping = True
                    if self._queue.empty():
                        return
                    continue
                fn, future = item
                try:
                    result = await fn()
                    if not future.done():
                        future.set_result(result)
                except asyncio.CancelledError:
                    # worker 被强制取消（stop 超时/外部 cancel）：
                    # 在途操作的 Future 必须显式失败，调用方不得永久悬挂
                    if not future.done():
                        future.set_exception(
                            RuntimeError(f"WriteQueue[{self.name}] 已停止，操作未执行")
                        )
                    raise
                except Exception as e:  # noqa: BLE001 - 单笔失败不拖垮 worker
                    logger.error(f"WriteQueue[{self.name}] 操作失败: {e}")
                    if not future.done():
                        future.set_exception(e)
            finally:
                self._queue.task_done()


class _suppress_cancel:
    """asyncio.CancelledError 的 contextlib.suppress 替身（3.11+ 它继承 BaseException）"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, asyncio.CancelledError)
