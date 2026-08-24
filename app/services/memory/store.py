"""长期记忆 - sqlite 存储层

设计要点：
- sqlite3 标准库同步实现，WAL 模式（读写并发友好）；连接带 threading.Lock
- 软删除：deleted_at 置时间戳，默认查询一律过滤
- 向量以 float32 BLOB 存储（紧凑）；读取时还原为 list[float]
- 本层不做任何打分/业务逻辑，只做 CRUD —— 打分在 scoring.py，编排在 service.py
- 所有方法为同步阻塞；由上层 service 通过 to_thread / 写队列调度
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.services.memory.types import MemoryItem, MemoryType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.3,
    embedding BLOB,
    user_id TEXT NOT NULL DEFAULT 'local',
    session_id TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    deleted_at REAL,
    consolidated_into TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories (user_id, type, deleted_at);
"""


def _pack_embedding(vec: list[float] | None) -> bytes | None:
    """向量 → float32 小端 BLOB"""
    if vec is None:
        return None
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    """float32 小端 BLOB → 向量；空值安全"""
    if not blob:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    try:
        metadata = json.loads(row["metadata"] or "{}")
        if not isinstance(metadata, dict):
            metadata = {"_raw": str(metadata)}
    except json.JSONDecodeError:
        metadata = {}  # 坏数据容错：读取路径永不因脏 metadata 崩溃
    try:
        importance = float(row["importance"])
    except (TypeError, ValueError):
        # REAL 亲和列仍可能被写入非数值 TEXT（迁移/手改库）；
        # 边界容错同时保护 recall / consolidate / to_dict 三处消费方
        logger.warning(f"记忆行 {row['id']!r} importance 非数值({row['importance']!r})，回退默认 0.3")
        importance = 0.3
    return MemoryItem(
        id=row["id"],
        type=MemoryType(row["type"]),
        content=row["content"],
        importance=importance,
        embedding=_unpack_embedding(row["embedding"]),
        user_id=row["user_id"],
        session_id=row["session_id"],
        metadata=metadata,
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        deleted_at=row["deleted_at"],
        consolidated_into=row["consolidated_into"],
    )


def _rows_to_items(rows: list[sqlite3.Row]) -> list[MemoryItem]:
    """逐行容错转换：单行脏数据（如未知 type 值）跳过并告警，
    不让一行坏数据使整批查询持续失败"""
    items: list[MemoryItem] = []
    for r in rows:
        try:
            items.append(_row_to_item(r))
        except Exception as e:  # noqa: BLE001 - 读取路径抗脏数据
            logger.warning(f"跳过损坏记忆行 id={r['id']} type={r['type']!r}: {e}")
    return items


class MemoryStore:
    """sqlite 长期记忆存储（同步 API）"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
        except BaseException:
            # 半途失败（库损坏/权限/磁盘满）先关连接再抛，
            # 否则 _ensure_store 每次重试都会泄漏一个句柄
            self._conn.close()
            raise
        logger.info(f"MemoryStore 就绪: {db_path}")

    # ──────────────── 写入 ────────────────

    def add(self, item: MemoryItem) -> str:
        """插入一条记忆（同 id 幂等覆盖）"""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, type, content, importance, embedding, user_id, session_id,
                 metadata, created_at, last_accessed_at, access_count,
                 deleted_at, consolidated_into)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.type.value,
                    item.content,
                    item.importance,
                    _pack_embedding(item.embedding),
                    item.user_id,
                    item.session_id,
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.created_at,
                    item.last_accessed_at,
                    item.access_count,
                    item.deleted_at,
                    item.consolidated_into,
                ),
            )
            self._conn.commit()
        return item.id

    def touch(self, ids: list[str], at: float | None = None) -> int:
        """批量记录召回访问（更新 last_accessed_at 与 access_count）"""
        if not ids:
            return 0
        now = time.time() if at is None else at
        with self._lock:
            cur = self._conn.executemany(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id = ?
                """,
                [(now, mid) for mid in ids],
            )
            self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def soft_delete(self, memory_id: str, at: float | None = None) -> bool:
        """软删除单条"""
        now = time.time() if at is None else at
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, memory_id),
            )
            self._conn.commit()
        return bool(cur.rowcount)

    def soft_delete_by_user(self, user_id: str, at: float | None = None) -> int:
        """软删除某用户的全部记忆（API DELETE 语义），返回条数"""
        now = time.time() if at is None else at
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET deleted_at = ? WHERE user_id = ? AND deleted_at IS NULL",
                (now, user_id),
            )
            self._conn.commit()
        return cur.rowcount or 0

    def mark_consolidated(self, member_ids: list[str], semantic_id: str) -> int:
        """情景记忆被巩固后：软删成员并回填指向语义记忆的引用"""
        if not member_ids:
            return 0
        now = time.time()
        with self._lock:
            cur = self._conn.executemany(
                """
                UPDATE memories
                SET deleted_at = ?, consolidated_into = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                [(now, semantic_id, mid) for mid in member_ids],
            )
            self._conn.commit()
        return cur.rowcount or 0

    # ──────────────── 读取 ────────────────

    def get(self, memory_id: str, *, include_deleted: bool = False) -> MemoryItem | None:
        """按 ID 读取单条（行损坏时与查无此行同语义：None）"""
        query = "SELECT * FROM memories WHERE id = ?"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        with self._lock:
            row = self._conn.execute(query, (memory_id,)).fetchone()
        items = _rows_to_items([row]) if row is not None else []
        return items[0] if items else None

    def candidates(
        self,
        *,
        user_id: str | None = None,
        types: list[MemoryType] | None = None,
    ) -> list[MemoryItem]:
        """召回候选集：全部未删除、未巩固消亡的记忆（含无向量者）

        consolidation 后的情景成员已软删除，天然不进候选。
        """
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            params.extend(t.value for t in types)
        query = f"SELECT * FROM memories WHERE {' AND '.join(clauses)}"  # noqa: S608
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return _rows_to_items(rows)

    def list_items(
        self,
        *,
        user_id: str | None = None,
        types: list[MemoryType] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """按条件列举（新→旧），供 API 与调试"""
        clauses: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            params.extend(t.value for t in types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ?"  # noqa: S608
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return _rows_to_items(rows)

    def stats(self) -> dict[str, Any]:
        """全局统计（按类型计数 + 删除数），供健康检查/API

        语义：total=全部行数，active=存活（未软删除），deleted=已软删除。
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT type,
                       COUNT(*) AS total,
                       SUM(CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN deleted_at IS NOT NULL THEN 1 ELSE 0 END) AS deleted
                FROM memories GROUP BY type
                """
            ).fetchall()
        by_type = {
            r["type"]: {
                "total": r["total"],
                "active": r["active"] or 0,
                "deleted": r["deleted"] or 0,
            }
            for r in rows
        }
        return {"by_type": by_type}

    def close(self) -> None:
        """关闭连接（WAL checkpoint 收尾）"""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as e:  # pragma: no cover - 关闭路径尽力而为
                logger.warning(f"WAL checkpoint 失败: {e}")
            self._conn.close()
