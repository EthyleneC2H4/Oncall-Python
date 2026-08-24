"""MemoryService 门面单测：写入/召回/巩固/遗忘/降级路径（全部注入替身，零外部依赖）"""

import asyncio
from types import SimpleNamespace

import pytest

from app.config import config
from app.services.memory import MemoryType
from app.services.memory.queue import WriteQueue
from app.services.memory.service import MemoryService, _mean_vector, _renormalize
from app.services.memory.store import MemoryStore
from app.services.memory.types import MemoryItem


class FakeEmbedder:
    """确定性假嵌入器：text → 预设向量；可注入失败"""

    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.default = [1.0, 0.0]
        self.fail = False
        self.calls: list[str] = []

    def embed_query_safe(self, text: str) -> list[float] | None:
        self.calls.append(text)
        if self.fail:
            return None
        return self.vectors.get(text, self.default)


@pytest.fixture
def memory(tmp_path):
    """注入假嵌入器 + 临时库的服务实例（写队列直通模式）"""
    embedder = FakeEmbedder()
    store = MemoryStore(db_path=str(tmp_path / "mem.db"))
    svc = MemoryService(store=store, embedder=embedder)
    yield SimpleNamespace(svc=svc, embedder=embedder, store=store)
    store.close()


def _candidates_contains(store: MemoryStore, memory_id: str) -> bool:
    return any(m.id == memory_id for m in store.candidates())


class TestEnabledSwitch:
    async def test_disabled_all_ops_noop(self, monkeypatch):
        monkeypatch.setattr(config, "memory_enabled", False)
        svc = MemoryService(embedder=FakeEmbedder())
        assert await svc.write_episodic("内容") == ""
        assert await svc.recall("查询") == []
        assert await svc.forget_user("u") == 0
        assert await svc.list_items() == []
        assert svc._store is None  # 永不触库

    async def test_enabled_by_default(self, memory):
        assert memory.svc.enabled is True


class TestWritePath:
    async def test_write_embeds_and_persists(self, memory):
        memory.embedder.vectors["内存泄漏诊断"] = [0.0, 1.0]
        memory_id = await memory.svc.write_episodic("内存泄漏诊断", session_id="s1")
        assert memory_id
        item = memory.store.get(memory_id)
        assert item.embedding == [0.0, 1.0]
        assert item.session_id == "s1"
        assert item.type is MemoryType.EPISODIC

    async def test_semantic_and_procedural_types(self, memory):
        sem_id = await memory.svc.write_semantic("经验总结", importance=0.7)
        proc_id = await memory.svc.write_procedural("处置步骤", importance=0.8)
        assert memory.store.get(sem_id).type is MemoryType.SEMANTIC
        assert memory.store.get(proc_id).type is MemoryType.PROCEDURAL

    async def test_embed_failure_still_persists_text(self, memory):
        """嵌入失败不阻断写入：无向量记忆仍可凭重要性被召回"""
        memory.embedder.fail = True
        mid = await memory.svc.write_episodic("重要事件", importance=0.9)
        assert memory.store.get(mid).embedding is None


class TestRecall:
    async def test_recall_ranks_relevant_first(self, memory):
        cpu_item = MemoryItem(
            type=MemoryType.EPISODIC,
            content="CPU 告警处理记录",
            importance=0.5,
            embedding=[1.0, 0.0],
        )
        disk_item = MemoryItem(
            type=MemoryType.EPISODIC,
            content="磁盘空间告警",
            importance=0.5,
            embedding=[0.0, 1.0],
        )
        memory.store.add(cpu_item)
        memory.store.add(disk_item)
        memory.embedder.default = [1.0, 0.0]  # 查询向量与 cpu 对齐

        hits = await memory.svc.recall("cpu 使用率过高")
        assert hits, "应至少命中一条"
        assert hits[0].id == cpu_item.id

    async def test_min_importance_filter(self, memory, monkeypatch):
        monkeypatch.setattr(config, "memory_min_importance", 0.5)
        memory.store.add(MemoryItem(type=MemoryType.EPISODIC, content="低价值闲聊", importance=0.1))
        memory.store.add(
            MemoryItem(type=MemoryType.EPISODIC, content="重大事故复盘", importance=0.8)
        )
        hits = await memory.svc.recall("任意查询")
        assert [h.content for h in hits] == ["重大事故复盘"]

    async def test_type_filter(self, memory):
        memory.store.add(MemoryItem(type=MemoryType.EPISODIC, content="情景"))
        memory.store.add(MemoryItem(type=MemoryType.SEMANTIC, content="语义"))
        hits = await memory.svc.recall("查询", types=[MemoryType.SEMANTIC])
        assert [h.type for h in hits] == [MemoryType.SEMANTIC]

    async def test_k_limit(self, memory):
        for i in range(10):
            memory.store.add(
                MemoryItem(type=MemoryType.EPISODIC, content=f"条目{i}", importance=0.9)
            )
        hits = await memory.svc.recall("查询", k=3)
        assert len(hits) == 3

    async def test_never_raises_on_store_failure(self, monkeypatch):
        """召回链路任何异常只降级为空列表"""

        def _boom():
            raise RuntimeError("库炸了")

        svc = MemoryService(embedder=FakeEmbedder())
        monkeypatch.setattr(svc, "_ensure_store", _boom)
        assert await svc.recall("查询") == []

    async def test_touch_updates_access_stats(self, memory):
        item = MemoryItem(type=MemoryType.EPISODIC, content="被召回者", importance=0.9)
        memory.store.add(item)
        hits = await memory.svc.recall("查询")
        assert hits
        # 触达统计走 fire-and-forget —— 轮询等待落库（上限 1s）
        for _ in range(100):
            if memory.store.get(item.id).access_count == 1:
                break
            await asyncio.sleep(0.01)
        assert memory.store.get(item.id).access_count == 1


class TestConsolidate:
    async def test_similar_episodics_merge_to_semantic(self, memory):
        near_b = [0.984807753, 0.173648178]  # 与 [1,0] 夹角 10°，cos ≈ 0.985 ≥ 0.85
        a = MemoryItem(
            type=MemoryType.EPISODIC, content="OOM 事故 A", embedding=[1.0, 0.0], importance=0.6
        )
        b = MemoryItem(
            type=MemoryType.EPISODIC, content="OOM 事故 B", embedding=near_b, importance=0.4
        )
        far = MemoryItem(
            type=MemoryType.EPISODIC, content="无关磁盘事件", embedding=[0.0, 1.0], importance=0.5
        )
        for m in (a, b, far):
            memory.store.add(m)

        stats = await memory.svc.consolidate()

        assert stats["clusters"] == 1
        assert stats["members_consolidated"] == 2
        semantic = memory.store.get(stats["semantic_ids"][0])
        assert semantic.type is MemoryType.SEMANTIC
        assert semantic.importance == 0.6  # 取成员最大值
        assert set(semantic.metadata["consolidated_from"]) == {a.id, b.id}
        assert "OOM 事故 A" in semantic.content and "OOM 事故 B" in semantic.content
        # 成员已软删除、不再进候选；孤立情景保持原样
        assert memory.store.get(a.id) is None
        assert not _candidates_contains(memory.store, a.id)
        assert _candidates_contains(memory.store, far.id)

    async def test_consolidation_skipped_when_no_vectors(self, memory):
        memory.embedder.fail = True
        await memory.svc.write_episodic("无向量A")
        await memory.svc.write_episodic("无向量B")
        stats = await memory.svc.consolidate()
        assert stats["clusters"] == 0

    async def test_threshold_parameter_controls_cluster(self, memory):
        orthogonal_pair = (
            MemoryItem(type=MemoryType.EPISODIC, content="X", embedding=[1.0, 0.0]),
            MemoryItem(type=MemoryType.EPISODIC, content="Y", embedding=[0.0, 1.0]),
        )
        for m in orthogonal_pair:
            memory.store.add(m)
        stats = await memory.svc.consolidate(threshold=0.5)  # 正交 cos=0 < 0.5 → 不合并
        assert stats["clusters"] == 0


class TestForgetAndStats:
    async def test_forget_user_soft_deletes(self, memory):
        await memory.svc.write_episodic("a", user_id="alice")
        await memory.svc.write_episodic("b", user_id="alice")
        await memory.svc.write_episodic("c", user_id="bob")
        deleted = await memory.svc.forget_user("alice")
        assert deleted == 2
        remaining = await memory.svc.list_items()
        assert [i.user_id for i in remaining] == ["bob"]

    async def test_stats_shape(self, memory):
        stats = await memory.svc.stats()
        assert stats["enabled"] is True
        assert "by_type" in stats


class TestVectorMathHelpers:
    def test_mean_vector(self):
        assert _mean_vector([[1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]

    def test_renormalize_zero_vector_safe(self):
        assert _renormalize([0.0, 0.0]) == [0.0, 0.0]

    def test_renormalize_unit_length(self):
        out = _renormalize([3.0, 4.0])
        assert abs(sum(x * x for x in out) - 1.0) < 1e-12


class TestAdversarialRegression:
    """对抗评审确认问题的回归测试（门面失败安全 / 停止语义 / 脏数据容错）"""

    async def test_write_never_raises(self, memory, monkeypatch):
        """回归 #4：写入链路任何异常降级为空 id，绝不向调用方抛出"""

        async def _boom(*a, **k):
            raise RuntimeError("队列炸了")

        monkeypatch.setattr(memory.svc, "_submit_store", _boom)
        assert await memory.svc.write_episodic("内容") == ""
        assert await memory.svc.write_semantic("经验") == ""

    async def test_forget_user_never_raises(self, memory, monkeypatch):
        """回归 #4：遗忘失败安全返回 0"""
        async def _boom(*a, **k):
            raise RuntimeError("删除失败")

        monkeypatch.setattr(memory.svc, "_submit_store", _boom)
        assert await memory.svc.forget_user("alice") == 0

    async def test_consolidate_never_raises(self, memory, monkeypatch):
        """回归 #4：巩固失败返回零统计而非异常"""

        def _boom(*a, **k):
            raise RuntimeError("候选查询失败")

        monkeypatch.setattr(memory.store, "candidates", _boom)
        stats = await memory.svc.consolidate()
        assert stats == {"clusters": 0, "members_consolidated": 0, "semantic_ids": []}

    async def test_recall_dim_mismatch_skips_stale_vectors(self, memory):
        """回归 #3：换嵌入模型后维度失配的存量向量被跳过并告警，
        不再静默按相关性 0 参与排序（否则会凭重要性注入无关记忆）"""
        memory.embedder.default = [1.0, 0.0]  # 新模型 2 维
        stale_3dim = MemoryItem(  # 存量 3 维向量（旧模型）
            type=MemoryType.EPISODIC, content="陈年旧事", importance=0.95, embedding=[1, 0, 0]
        )
        current = MemoryItem(
            type=MemoryType.EPISODIC, content="当前经验", importance=0.5, embedding=[1.0, 0.0]
        )
        memory.store.add(stale_3dim)
        memory.store.add(current)
        hits = await memory.svc.recall("查询")
        # 高重要度但维度失配者不得入选；同维相关者正常召回
        assert [h.id for h in hits] == [current.id]

    async def test_recall_dim_mismatch_degrades_to_empty_not_raise(self, memory):
        """全部失配时返回空列表（宁可少召回，不注垃圾、不崩溃）"""
        memory.embedder.default = [1.0, 0.0]
        memory.store.add(
            MemoryItem(type=MemoryType.EPISODIC, content="旧维", importance=0.9, embedding=[1, 0, 0])
        )
        assert await memory.svc.recall("查询") == []

    async def test_stop_blocks_lazy_reconnect(self, tmp_path):
        """回归 #5：stop() 后调用不触发惰性重建（零副作用，与 disabled 一致）"""
        svc = MemoryService(store=MemoryStore(db_path=str(tmp_path / "m.db")), embedder=FakeEmbedder())
        assert await svc.start() is True
        await svc.stop()
        assert svc._store is None
        assert await svc.write_episodic("内容") == ""
        assert await svc.recall("查询") == []
        assert await svc.forget_user("u") == 0
        assert svc._store is None  # 未被重建

    async def test_start_disabled_returns_false(self, monkeypatch, tmp_path):
        """回归 #14/#15：disabled 时 start() 返回 False 且不建库"""
        monkeypatch.setattr(config, "memory_enabled", False)
        svc = MemoryService(embedder=FakeEmbedder())
        assert await svc.start() is False

    async def test_restart_after_stop_reconnects(self, tmp_path, monkeypatch):
        """start() 重置停止标志：支持同进程重启"""
        svc = MemoryService(store=None, embedder=FakeEmbedder(), queue=WriteQueue())
        monkeypatch.setattr(config, "memory_db_path", str(tmp_path / "restart.db"))
        assert await svc.start() is True
        await svc.stop()
        assert await svc.start() is True  # 二次启动成功（_stopped 已被重置）
        await svc.stop()
