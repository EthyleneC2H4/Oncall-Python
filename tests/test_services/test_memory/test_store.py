"""MemoryStore sqlite 层单测：CRUD / 软删除 / 巩固标记 / 向量序列化 / WAL"""

import pytest

from app.services.memory.store import MemoryStore, _pack_embedding, _unpack_embedding
from app.services.memory.types import MemoryItem, MemoryType


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "mem.db"))
    yield s
    s.close()


def _item(**overrides) -> MemoryItem:
    defaults: dict = {
        "type": MemoryType.EPISODIC,
        "content": "CPU 告警，内存泄漏",
        "importance": 0.5,
        "embedding": [0.1, 0.2, 0.3],
        "user_id": "u1",
        "session_id": "s1",
        "metadata": {"k": "v"},
    }
    defaults.update(overrides)
    return MemoryItem(**defaults)


class TestEmbeddingSerialization:
    def test_roundtrip_float32_tolerance(self):
        vec = [0.123456789, -0.987654321, 1.5, 0.0]
        restored = _unpack_embedding(_pack_embedding(vec))
        assert len(restored) == 4
        assert all(abs(a - b) < 1e-6 for a, b in zip(vec, restored, strict=True))

    def test_none_passthrough(self):
        assert _pack_embedding(None) is None
        assert _unpack_embedding(None) is None

    def test_empty_blob(self):
        assert _unpack_embedding(b"") is None


class TestCrud:
    def test_add_and_get(self, store):
        item = _item()
        store.add(item)
        loaded = store.get(item.id)
        assert loaded is not None
        assert loaded.content == "CPU 告警，内存泄漏"
        assert loaded.type is MemoryType.EPISODIC
        assert loaded.embedding is not None
        assert all(
            abs(a - b) < 1e-6
            for a, b in zip([0.1, 0.2, 0.3], loaded.embedding, strict=True)
        )
        assert loaded.metadata == {"k": "v"}

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_add_same_id_overwrites(self, store):
        item = _item()
        store.add(item)
        item.content = "更新后的内容"
        store.add(item)
        assert store.get(item.id).content == "更新后的内容"

    def test_list_items_newest_first(self, store):
        old = _item(content="旧", created_at=100.0)
        new = _item(content="新", created_at=200.0)
        store.add(old)
        store.add(new)
        listed = store.list_items(user_id="u1")
        assert [i.content for i in listed] == ["新", "旧"]


class TestSoftDelete:
    def test_soft_delete_hides_from_default_views(self, store):
        item = _item()
        store.add(item)
        assert store.soft_delete(item.id, at=500.0) is True
        assert store.get(item.id) is None  # 默认不可见
        assert store.candidates() == []
        assert store.list_items() == []
        # include_deleted 可恢复查看
        deleted = store.get(item.id, include_deleted=True)
        assert deleted.deleted_at == 500.0

    def test_double_delete_returns_false(self, store):
        item = _item()
        store.add(item)
        assert store.soft_delete(item.id) is True
        assert store.soft_delete(item.id) is False

    def test_delete_by_user(self, store):
        a, b, c = _item(), _item(), _item(user_id="u2")
        store.add(a)
        store.add(b)
        store.add(c)
        assert store.soft_delete_by_user("u1") == 2
        remaining = store.candidates()
        assert [i.user_id for i in remaining] == ["u2"]


class TestTouchAndConsolidation:
    def test_touch_increments(self, store):
        item = _item()
        store.add(item)
        store.touch([item.id], at=999.0)
        loaded = store.get(item.id)
        assert loaded.access_count == 1
        assert loaded.last_accessed_at == 999.0

    def test_touch_empty_noop(self, store):
        assert store.touch([]) == 0

    def test_mark_consolidated(self, store):
        m1, m2 = _item(), _item(embedding=None)
        store.add(m1)
        store.add(m2)
        count = store.mark_consolidated([m1.id, m2.id], semantic_id="sem-1")
        assert count == 2
        survivor = store.get(m1.id, include_deleted=True)
        assert survivor.deleted_at is not None
        assert survivor.consolidated_into == "sem-1"
        # 巩固后的成员不进召回候选
        assert store.candidates(types=[MemoryType.EPISODIC]) == []


class TestInfrastructure:
    def test_wal_mode_enabled(self, tmp_path):
        s = MemoryStore(db_path=str(tmp_path / "wal.db"))
        try:
            mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
        finally:
            s.close()

    def test_parent_dirs_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "mem.db"
        MemoryStore(db_path=str(nested)).close()
        assert nested.exists()

    def test_stats_by_type(self, store):
        store.add(_item())
        store.add(_item(type=MemoryType.SEMANTIC))
        dead = _item()
        store.add(dead)
        store.soft_delete(dead.id)
        stats = store.stats()
        # total=全部行数，active=存活，deleted=软删除
        assert stats["by_type"]["episodic"] == {"total": 2, "active": 1, "deleted": 1}
        assert stats["by_type"]["semantic"] == {"total": 1, "active": 1, "deleted": 0}

    def test_corrupt_metadata_tolerated(self, store):
        """metadata 非法 JSON / 非对象时读取不崩溃，降级为空/原始值"""
        item = _item(metadata={})
        store.add(item)
        for bad in ("{broken", "[1,2]", "null"):
            store._conn.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?", (bad, item.id)
            )
            store._conn.commit()
            loaded = store.get(item.id)
            assert loaded is not None, f"坏 metadata {bad!r} 不应阻断读取"
            assert isinstance(loaded.metadata, dict)
            assert loaded.content == item.content


class TestDirtyRowTolerance:
    """对抗评审 #6 回归：单行脏数据只跳过该行，不拖垮整批查询"""

    def _poison_type(self, store: MemoryStore, memory_id: str) -> None:
        store._conn.execute(
            "UPDATE memories SET type = 'alien' WHERE id = ?", (memory_id,)
        )
        store._conn.commit()

    def test_unknown_type_row_skipped_in_batch_reads(self, store):
        good_a, bad, good_b = _item(content="好A"), _item(content="坏行"), _item(content="好B")
        for m in (good_a, bad, good_b):
            store.add(m)
        self._poison_type(store, bad.id)

        candidates = store.candidates()
        assert sorted(m.content for m in candidates) == ["好A", "好B"]
        listed = store.list_items()
        assert sorted(m.content for m in listed) == ["好A", "好B"]

    def test_single_unknown_type_get_returns_none(self, store):
        item = _item()
        store.add(item)
        self._poison_type(store, item.id)
        assert store.get(item.id) is None  # 脏行走单行容错路径

    def test_non_numeric_importance_falls_back(self, store):
        """REAL 亲和列仍可能被写入非数值 TEXT（迁移/手改库）→ 回退默认 0.3"""
        item = _item(importance=0.9)
        store.add(item)
        store._conn.execute(
            "UPDATE memories SET importance = 'oops' WHERE id = ?", (item.id,)
        )
        store._conn.commit()
        loaded = store.get(item.id)
        assert loaded is not None
        assert loaded.importance == 0.3
