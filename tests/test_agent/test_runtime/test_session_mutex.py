"""会话互斥测试

回归点（审计发现）：同 session 并发 run 会交错写 MemorySaver 检查点——
ReAct 串话、PlanExecute 的 delete_thread 清掉在途追加通道状态。
覆盖：互斥原语本身 + PlanExecuteRuntime 集成（同会话串行/跨会话并行）。
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.runtime.events import EventType
from app.agent.runtime.plan_execute_runtime import (
    NODE_PLANNER,
    NODE_REPLANNER,
    PlanExecuteRuntime,
)
from app.agent.runtime.session_mutex import session_mutex


class TestMutexPrimitive:
    @pytest.mark.asyncio
    async def test_same_key_critical_sections_do_not_overlap(self):
        order: list[str] = []
        locks: dict[str, asyncio.Lock] = {}

        async def worker(tag: str) -> None:
            async with session_mutex(locks, "s1"):
                order.append(f"enter:{tag}")
                await asyncio.sleep(0.02)  # 让出事件循环，制造交叠窗口
                order.append(f"exit:{tag}")

        await asyncio.gather(worker("A"), worker("B"))

        # enter/exit 必须成对相邻：临界区不重叠
        for tag in ("A", "B"):
            assert order.index(f"exit:{tag}") == order.index(f"enter:{tag}") + 1

    @pytest.mark.asyncio
    async def test_different_keys_run_concurrently(self):
        locks: dict[str, asyncio.Lock] = {}
        b_entered = asyncio.Event()
        order: list[str] = []

        async def worker_a() -> None:
            async with session_mutex(locks, "s1"):
                order.append("a-enter")
                await b_entered.wait()  # 等 B 进临界区；若跨 key 互斥则此处死锁
                order.append("a-exit")

        async def worker_b() -> None:
            await asyncio.sleep(0.01)  # 确保 A 先进入
            async with session_mutex(locks, "s2"):
                b_entered.set()

        await asyncio.wait_for(asyncio.gather(worker_a(), worker_b()), timeout=2.0)
        assert order == ["a-enter", "a-exit"]


class ConcurrencyTrackingGraph:
    """记录 astream 并发度的假图（同刻活跃生成器数）"""

    def __init__(self):
        self.active = 0
        self.max_active = 0

    def astream(self, input=None, config=None, stream_mode=None):  # noqa: A002
        graph = self

        async def _gen():
            graph.active += 1
            graph.max_active = max(graph.max_active, graph.active)
            await asyncio.sleep(0.03)
            yield {NODE_PLANNER: {"plan": ["步骤"]}}
            await asyncio.sleep(0.03)
            yield {NODE_REPLANNER: {"response": "# 报告", "plan": []}}
            graph.active -= 1

        return _gen()

    def get_state(self, config=None):
        return SimpleNamespace(values={"response": "# 报告"})


def _make_runtime(graph: Any) -> PlanExecuteRuntime:
    rt = PlanExecuteRuntime.__new__(PlanExecuteRuntime)  # 跳过真实图构建
    rt.checkpointer = None
    rt.graph = graph
    rt._session_locks = {}
    return rt


async def _drain(runtime: PlanExecuteRuntime, task: str, session_id: str) -> list:
    return [e async for e in runtime.run(task, session_id=session_id)]


class TestPlanExecuteSessionSerialization:
    @pytest.mark.asyncio
    async def test_same_session_runs_serialized(self):
        graph = ConcurrencyTrackingGraph()
        rt = _make_runtime(graph)

        results = await asyncio.gather(
            _drain(rt, "任务一", "same-session"),
            _drain(rt, "任务二", "same-session"),
        )

        # 两次运行都完整收尾，且图从未被并发进入
        for events in results:
            assert [e.type for e in events][-1] is EventType.COMPLETE
        assert graph.max_active == 1

    @pytest.mark.asyncio
    async def test_different_sessions_run_concurrently(self):
        graph = ConcurrencyTrackingGraph()
        rt = _make_runtime(graph)

        await asyncio.gather(
            _drain(rt, "任务一", "session-a"),
            _drain(rt, "任务二", "session-b"),
        )

        assert graph.max_active == 2  # 跨会话不互相阻塞
