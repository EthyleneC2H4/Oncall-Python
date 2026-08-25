"""会话级互斥：同一 session_id 的并发 run 串行化

双击重试、流式未断开又发新请求是常见 UX——两条 run 会同时对同一
thread_id 执行 superstep，检查点按时间戳交替入栈：
- ReAct 表现为对话历史串话、工具调用与回答错位；
- plan_execute 更糟：第二次 run 启动时的 delete_thread 会清掉第一次的
  在途追加通道状态，产出空回答却标记「任务执行完成」。

锁对象恒定大小（<200B/会话），量级远小于其保护的检查点 blob，
不做淘汰——淘汰会引入「取旧锁 vs 建新锁」的互斥间隙。
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def session_mutex(locks: dict[str, asyncio.Lock], session_id: str) -> AsyncIterator[None]:
    """按 session_id 取锁并串行化临界区（dict 由各运行时实例持有）"""
    lock = locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        yield
