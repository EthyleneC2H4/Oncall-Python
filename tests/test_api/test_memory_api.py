"""长期记忆 API 测试：GET / DELETE /api/memory/{user_id}（隔离临时库）"""

import pytest

from app.services.memory import MemoryService, memory_service
from app.services.memory.store import MemoryStore
from app.services.memory.types import MemoryItem, MemoryType


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """把全局单例指向临时库，测试后恢复（防止污染 data/memory.db）"""
    saved_store = memory_service._store
    temp_store = MemoryStore(db_path=str(tmp_path / "api-mem.db"))
    monkeypatch.setattr(memory_service, "_store", temp_store)
    yield memory_service
    temp_store.close()
    monkeypatch.setattr(memory_service, "_store", saved_store)


def _seed(store: MemoryStore, **overrides) -> MemoryItem:
    defaults: dict = {
        "type": MemoryType.EPISODIC,
        "content": "内容",
        "user_id": "u1",
    }
    defaults.update(overrides)
    item = MemoryItem(**defaults)
    store.add(item)
    return item


class TestGetUserMemory:
    def test_empty_user_returns_zero(self, test_app, isolated_memory):
        resp = test_app.get("/api/memory/local")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 200
        assert data["data"]["user_id"] == "local"
        assert data["data"]["total"] == 0
        assert data["data"]["memories"] == []
        assert "by_type" in data["data"]["stats"]

    def test_list_after_writes_scoped_by_user(self, test_app, isolated_memory):
        store = isolated_memory._store
        _seed(store, content="情景记忆", user_id="alice", type=MemoryType.EPISODIC)
        _seed(store, content="语义经验", user_id="alice", type=MemoryType.SEMANTIC)
        _seed(store, content="别人的", user_id="bob")

        resp = test_app.get("/api/memory/alice")
        data = resp.json()["data"]
        assert data["total"] == 2
        assert {m["content"] for m in data["memories"]} == {"情景记忆", "语义经验"}
        # 序列化不泄漏向量字段
        assert all("embedding" not in m for m in data["memories"])

    def test_type_filter_param(self, test_app, isolated_memory):
        store = isolated_memory._store
        _seed(store, content="e", user_id="u", type=MemoryType.EPISODIC)
        _seed(store, content="p", user_id="u", type=MemoryType.PROCEDURAL)

        resp = test_app.get("/api/memory/u", params={"type": "procedural"})
        memories = resp.json()["data"]["memories"]
        assert [m["type"] for m in memories] == ["procedural"]

    def test_invalid_type_returns_400(self, test_app, isolated_memory):
        resp = test_app.get("/api/memory/u", params={"type": "not-a-type"})
        assert resp.status_code == 400


class TestDeleteUserMemory:
    def test_delete_removes_all_for_user(self, test_app, isolated_memory):
        store = isolated_memory._store
        _seed(store, content="a", user_id="carol")
        _seed(store, content="b", user_id="carol", type=MemoryType.SEMANTIC)

        resp = test_app.delete("/api/memory/carol")
        data = resp.json()
        assert data["code"] == 200
        assert data["data"]["deleted"] == 2

        after = test_app.get("/api/memory/carol").json()["data"]
        assert after["total"] == 0

    def test_delete_unknown_user_returns_zero(self, test_app, isolated_memory):
        resp = test_app.delete("/api/memory/ghost")
        data = resp.json()
        assert data["code"] == 200
        assert data["data"]["deleted"] == 0


class TestServiceWiring:
    def test_singleton_is_memory_service(self):
        """API 层与 lifespan 引用的是同一个 MemoryService 单例"""
        assert isinstance(memory_service, MemoryService)
