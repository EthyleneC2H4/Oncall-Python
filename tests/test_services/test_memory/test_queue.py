"""WriteQueue 单 worker 写队列单测：FIFO 顺序 / 异常隔离 / 优雅停止 / 直通模式
/ 停止语义（Future 清算、取消传播、哨兵不毒化重启）"""

import asyncio
import contextlib

import pytest

from app.services.memory.queue import WriteQueue


class TestInlinePassthrough:
    async def test_unstarted_queue_executes_inline(self):
        """未 start 时 submit 直通执行（测试同步友好）"""
        queue = WriteQueue()
        assert not queue.running
        result = await queue.submit(_make_op("直接执行"))
        assert result == "直接执行"


class TestFifoOrdering:
    async def test_fifo_execution_order(self):
        queue = WriteQueue()
        queue.start()
        order: list[int] = []

        async def _record(i: int, delay: float) -> int:
            await asyncio.sleep(delay)
            order.append(i)
            return i

        # 先提交的反而睡得更久 —— 仍必须按提交顺序完成（单 worker 串行）
        futures = [
            queue.submit(lambda i=i, d=d: _record(i, d))
            for i, d in [(1, 0.03), (2, 0.01), (3, 0.0)]
        ]
        results = [await f for f in futures]
        await queue.stop()

        assert results == [1, 2, 3]
        assert order == [1, 2, 3]

    async def test_drain_waits_for_backlog(self):
        queue = WriteQueue()
        queue.start()
        done: list[str] = []

        for i in range(5):
            queue.submit(lambda i=i: _append(done, f"op-{i}"))
        await queue.drain()
        assert len(done) == 5
        await queue.stop()


async def _append(target: list[str], value: str) -> None:
    target.append(value)


def _make_op(value: str):
    async def _op() -> str:
        return value

    return _op


class TestErrorIsolation:
    async def test_failing_op_does_not_kill_worker(self):
        queue = WriteQueue()
        queue.start()

        async def _boom() -> None:
            raise RuntimeError("写入失败")

        failed = queue.submit(_boom)
        survivor = queue.submit(_make_op("后续任务"))

        with pytest.raises(RuntimeError, match="写入失败"):
            await failed
        assert await survivor == "后续任务"
        assert queue.running  # worker 存活
        await queue.stop()


class TestLifecycle:
    async def test_stop_drains_pending_then_exits(self):
        queue = WriteQueue()
        queue.start()
        done: list[str] = []
        for i in range(3):
            queue.submit(lambda i=i: _append(done, f"t{i}"))
        await queue.stop()  # stop 内部先排空
        assert sorted(done) == ["t0", "t1", "t2"]
        assert not queue.running

    async def test_start_is_idempotent(self):
        queue = WriteQueue()
        queue.start()
        worker_first = queue._worker
        queue.start()
        assert queue._worker is worker_first
        await queue.stop()

    async def test_submit_after_stop_is_inline(self):
        queue = WriteQueue()
        queue.start()
        await queue.stop()
        # 停止后回退直通模式，调用方不会悬挂
        assert await queue.submit(_make_op("停止后")) == "停止后"


class TestStopSemantics:
    """对抗评审 #8/#9 回归：停止后所有 Future 必然 resolve、取消不被吞"""

    async def test_stop_resolves_every_submitted_future(self):
        """回归 #8：stop() 返回后，任何未执行的 Future 必须已失败而非永久悬挂"""
        queue = WriteQueue()
        queue.start()

        gate = asyncio.Event()

        async def _blocked() -> None:
            await gate.wait()  # 卡住 worker，制造 stop 时仍在队列中的积压

        blocked_future = queue.submit(_blocked)
        backlog_futures = [queue.submit(_make_op(f"op-{i}")) for i in range(3)]

        # 不解锁 gate：stop 超时 → 强制取消路径，积压 Future 被 fail-fast 清算
        await asyncio.wait_for(queue.stop(), timeout=10.0)

        with pytest.raises(RuntimeError, match="已停止"):
            await asyncio.wait_for(asyncio.shield(blocked_future), timeout=1.0)
        for f in backlog_futures:
            try:
                await asyncio.wait_for(asyncio.shield(f), timeout=1.0)
            except RuntimeError as e:
                assert "已停止" in str(e)  # 未执行者必须显式失败
            except TimeoutError:  # pragma: no cover - 不允许悬挂
                pytest.fail("Future 在 stop 后仍悬挂")
        assert not queue.running

    async def test_worker_cancelled_error_propagates_from_stop(self):
        """回归 #9：外部 cancel 传播出 stop()（不吞取消、不清 worker 引用）"""
        queue = WriteQueue()
        queue.start()

        async def _never() -> None:
            await asyncio.Event().wait()

        queue.submit(_never)
        stop_task = asyncio.ensure_future(queue.stop())
        await asyncio.sleep(0.01)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
        # worker 引用保留（可能仍在跑），running 状态由其真实生命周期决定
        assert queue._worker is not None
        # 测试收尾：清理仍阻塞的 worker
        queue._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queue._worker

    async def test_stale_sentinel_does_not_poison_restart(self):
        """陈旧哨兵被丢弃：stop→start 循环后队列依然可用"""
        queue = WriteQueue()
        queue.start()
        first = await queue.submit(_make_op("第一轮"))
        assert first == "第一轮"
        await queue.stop()

        queue.start()
        result = await queue.submit(_make_op("第二轮"))
        await queue.drain()
        assert result == "第二轮"
        assert queue.running
        await queue.stop()
