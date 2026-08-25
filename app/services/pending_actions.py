"""高风险待审动作存储

guard 拦截到 requires_confirmation 的工具调用时不执行，而是生成一条
pending action 落库并返回 action_id；人工经 POST /api/actions/{id}/approve|reject
裁决后，approve 侧从库中取回参数补执行。

sqlite 单文件存储（与长期记忆同一轻量选型）；状态机：
    pending → approved | rejected
超时未裁决的记录按 TTL 判过期（读取时惰性判定 + 清理）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ActionStatus(StrEnum):
    """待审动作状态"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    # 终态：已由 execute_approved 原子认领并（尝试）补执行——
    # 封死重复/并发 approve 导致的多次执行
    EXECUTED = "executed"


class PendingAction(BaseModel):
    """一条待审动作"""

    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""  # 触发确认门的原因（展示给审批人）
    session_id: str = ""
    request_id: str = ""
    status: ActionStatus = ActionStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    decided_at: float | None = None
    # approve 后回填的执行结果预览（观测用途）
    result_preview: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_actions (
    action_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    decided_at REAL,
    result_preview TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_actions (status, created_at);
"""


class PendingActionStore:
    """sqlite 待审动作存储（同步 API，FastAPI 侧经 to_thread 调用）"""

    def __init__(self, db_path: str = "data/pending_actions.db", ttl_seconds: float = 900.0):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        logger.info(f"PendingActionStore 就绪: {db_path} (ttl={ttl_seconds}s)")

    def propose(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
        reason: str = "",
        session_id: str = "",
        request_id: str = "",
    ) -> PendingAction:
        """登记一条新的待审动作"""
        action = PendingAction(
            tool_name=tool_name,
            args=args or {},
            reason=reason,
            session_id=session_id,
            request_id=request_id,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pending_actions
                (action_id, tool_name, args, reason, session_id, request_id,
                 status, created_at, decided_at, result_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.action_id,
                    action.tool_name,
                    json.dumps(action.args, ensure_ascii=False),
                    action.reason,
                    action.session_id,
                    action.request_id,
                    action.status.value,
                    action.created_at,
                    action.decided_at,
                    action.result_preview,
                ),
            )
            self._conn.commit()
        logger.info(f"待审动作已登记: {action.action_id} tool={tool_name}")
        return action

    def get(self, action_id: str) -> PendingAction | None:
        """按 id 读取（过期的 pending 自动标记 expired）"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
        if row is None:
            return None
        action = _row_to_action(row)
        if action.status is ActionStatus.PENDING and self._is_expired(action):
            return self._mark(action.action_id, ActionStatus.EXPIRED) or action
        return action

    def transition(
        self, action_id: str, from_status: ActionStatus, to_status: ActionStatus
    ) -> PendingAction | None:
        """原子状态转移：仅当当前状态为 from_status 时推进到 to_status

        返回转移后的最新记录；前置状态不符（含已被并发方抢先）返回 None。
        sqlite 单条条件 UPDATE 保证并发下的恰好一次语义。
        以 PENDING 为起点的转移额外带 TTL 条件：过期动作不允许被裁决，
        否则 15 分钟审批窗口只要没人调过 list_pending 就能无限延长。
        """
        with self._lock:
            sql = (
                "UPDATE pending_actions SET status = ?, decided_at = ? "
                "WHERE action_id = ? AND status = ?"
            )
            params: list[Any] = [to_status.value, time.time(), action_id, from_status.value]
            if from_status is ActionStatus.PENDING:
                sql += " AND created_at >= ?"
                params.append(time.time() - self.ttl_seconds)
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            if not cur.rowcount:
                return None
            row = self._conn.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
        return _row_to_action(row) if row else None

    def decide(self, action_id: str, decision: ActionStatus) -> PendingAction | None:
        """裁决：approved / rejected；仅 pending 可被裁决（幂等，重复裁决返回现状态）"""
        if decision not in (ActionStatus.APPROVED, ActionStatus.REJECTED):
            raise ValueError(f"非法裁决: {decision}")
        moved = self.transition(action_id, ActionStatus.PENDING, decision)
        if moved is not None:
            return moved
        # 未发生转移：不存在、已裁决或已过期——返回现状态，不二次改写
        return self.get(action_id)

    def attach_result(self, action_id: str, result_preview: str) -> None:
        """approve 执行完成后回填结果预览"""
        with self._lock:
            self._conn.execute(
                "UPDATE pending_actions SET result_preview = ? WHERE action_id = ?",
                (result_preview[:500], action_id),
            )
            self._conn.commit()

    def list_pending(self, *, include_expired: bool = False) -> list[PendingAction]:
        """列出待裁决动作（新→旧）"""
        self._sweep_expired()
        status = ActionStatus.EXPIRED if include_expired else ActionStatus.PENDING
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pending_actions WHERE status = ? ORDER BY created_at DESC LIMIT 100",
                (status.value,),
            ).fetchall()
        return [_row_to_action(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ──────────────── 内部 ────────────────

    def _is_expired(self, action: PendingAction) -> bool:
        return (time.time() - action.created_at) > self.ttl_seconds

    def _mark(self, action_id: str, status: ActionStatus) -> PendingAction | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE pending_actions SET status = ?, decided_at = ? WHERE action_id = ?",
                (status.value, time.time(), action_id),
            )
            self._conn.commit()
        if not cur.rowcount:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
        return _row_to_action(row) if row else None

    def _sweep_expired(self) -> int:
        """把超过 TTL 仍为 pending 的记录标为 expired（惰性清理）"""
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            cur = self._conn.execute(
                "UPDATE pending_actions SET status = ?, decided_at = ? "
                "WHERE status = 'pending' AND created_at < ?",
                (ActionStatus.EXPIRED.value, time.time(), cutoff),
            )
            self._conn.commit()
        if cur.rowcount:
            logger.info(f"清理过期待审动作 {cur.rowcount} 条")
        return cur.rowcount


def _row_to_action(row: sqlite3.Row) -> PendingAction:
    try:
        args = json.loads(row["args"] or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {"_raw": str(args)}
    try:
        status = ActionStatus(row["status"])
    except ValueError:
        status = ActionStatus.PENDING
    return PendingAction(
        action_id=row["action_id"],
        tool_name=row["tool_name"],
        args=args,
        reason=row["reason"],
        session_id=row["session_id"],
        request_id=row["request_id"],
        status=status,
        created_at=row["created_at"],
        decided_at=row["decided_at"],
        result_preview=row["result_preview"],
    )


# 全局单例（路径/TTL 从配置读取，惰性初始化见 guard.py）
_pending_action_store: PendingActionStore | None = None


def get_pending_action_store() -> PendingActionStore:
    """进程级单例访问器（首次调用时按配置建库）"""
    global _pending_action_store
    if _pending_action_store is None:
        from app.config import config

        _pending_action_store = PendingActionStore(
            db_path=config.pending_actions_db_path,
            ttl_seconds=config.pending_action_ttl_seconds,
        )
    return _pending_action_store


def reset_pending_action_store() -> None:
    """重置单例（测试用：换临时库路径后生效）"""
    global _pending_action_store
    if _pending_action_store is not None:
        try:
            _pending_action_store.close()
        except Exception:  # noqa: BLE001 - 测试清理尽力而为
            pass
    _pending_action_store = None
