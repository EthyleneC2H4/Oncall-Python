"""长期记忆 - 服务门面

组合 store（sqlite 持久化）+ queue（单 worker 写队列）+ scoring（纯函数打分）
+ 本地 BGE 向量（复用 P1.0 落地的 vector_embedding_service）。

对外能力：
- write_episodic / write_semantic / write_procedural：写入（异步排队，不阻塞事件循环）
- recall(query)：向量召回 + 加权打分 + 重要性下限过滤，永不向调用方抛异常
- consolidate()：情景记忆按余弦相似度贪心聚类 → 巩固为语义记忆（经验沉淀闭环）
- forget_user()：按用户软删除（API DELETE 语义）

memory_enabled=False 时所有操作为无副作用空操作 —— 关闭路径与无本模块行为一致。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from app.config import config
from app.services.memory.queue import WriteQueue
from app.services.memory.scoring import ScoreWeights, composite_score, cosine_similarity
from app.services.memory.store import MemoryStore
from app.services.memory.types import MemoryItem, MemoryType


class MemoryService:
    """长期记忆服务门面"""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        embedder: Any = None,
        queue: WriteQueue | None = None,
        weights: ScoreWeights | None = None,
    ):
        """初始化（惰性：sqlite 连接在首次真正使用时才建立）

        Args:
            store: 注入已有存储（测试用）；默认懒建 MemoryStore(config.memory_db_path)
            embedder: 注入嵌入器（测试用假向量）；默认全局 vector_embedding_service
            queue: 注入写队列；默认新建未启动的 WriteQueue（直通模式）
            weights: 打分权重；默认从配置读取
        """
        self._store_override = store
        self._store: MemoryStore | None = store
        self.embedder = embedder if embedder is not None else self._default_embedder()
        self.weights = weights
        self.write_queue = queue or WriteQueue()
        self._stopped = False  # stop() 后拒绝惰性重建（与 disabled 同款零副作用）
        self._bg: set[asyncio.Future[Any]] = set()  # 在途后台任务跟踪（关闭前等待落库）

    @staticmethod
    def _default_embedder() -> Any:
        """延迟导入全局嵌入器，避免模块导入即加载重依赖"""
        try:
            from app.services.vector_embedding_service import vector_embedding_service

            return vector_embedding_service
        except Exception as e:  # pragma: no cover - 导入失败极端场景
            logger.warning(f"加载默认嵌入器失败: {e}")
            return None

    # ──────────────── 开关与基础设施 ────────────────

    @property
    def enabled(self) -> bool:
        return bool(config.memory_enabled)

    def _ensure_store(self) -> MemoryStore | None:
        """惰性建库；disabled / 已停止 / 建库失败返回 None"""
        if not self.enabled or self._stopped:
            return None
        if self._store is None:
            try:
                self._store = MemoryStore(db_path=config.memory_db_path)
            except Exception as e:
                logger.error(f"MemoryStore 初始化失败，记忆功能降级关闭: {e}")
                return None
        return self._store

    def _weights(self) -> ScoreWeights:
        if self.weights is not None:
            return self.weights
        return ScoreWeights(
            relevance=config.memory_weight_relevance,
            importance=config.memory_weight_importance,
            recency=config.memory_weight_recency,
        )

    # ──────────────── 写入 ────────────────

    async def write(self, item: MemoryItem) -> str:
        """写入一条记忆（经写队列串行落库）

        失败安全：disabled/停止时返回 ""，存储层任何异常只记日志并返回 ""，
        绝不向调用方抛异常。
        """
        store = self._ensure_store()
        if store is None:
            return ""
        try:
            if item.embedding is None and self.embedder is not None:
                vec = await self._embed_safe(item.content)
                if vec:
                    item.embedding = vec
            memory_id: str = await self._submit_store("add", item)
            return memory_id
        except Exception as e:  # noqa: BLE001 - 门面级失败安全契约
            logger.warning(f"记忆写入失败（忽略）: {e}")
            return ""

    async def write_episodic(
        self,
        content: str,
        *,
        session_id: str = "",
        user_id: str = "local",
        importance: float = 0.3,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """记录一次具体交互（情景记忆）"""
        return await self.write(
            MemoryItem(
                type=MemoryType.EPISODIC,
                content=content,
                importance=importance,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        )

    async def write_semantic(
        self,
        content: str,
        *,
        user_id: str = "local",
        importance: float = 0.6,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """沉淀一条通用经验（语义记忆）"""
        return await self.write(
            MemoryItem(
                type=MemoryType.SEMANTIC,
                content=content,
                importance=importance,
                user_id=user_id,
                metadata=metadata or {},
            )
        )

    async def write_procedural(
        self,
        content: str,
        *,
        user_id: str = "local",
        importance: float = 0.6,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """沉淀一条处置方案（程序记忆）"""
        return await self.write(
            MemoryItem(
                type=MemoryType.PROCEDURAL,
                content=content,
                importance=importance,
                user_id=user_id,
                metadata=metadata or {},
            )
        )

    # ──────────────── 召回 ────────────────

    async def recall(
        self,
        query: str,
        *,
        k: int | None = None,
        types: list[MemoryType] | None = None,
        min_importance: float | None = None,
        user_id: str | None = None,
        now: float | None = None,
    ) -> list[MemoryItem]:
        """按查询召回相关记忆（打分排序取 Top-K）

        失败安全：任何异常只记日志并返回 []，绝不影响主流程。

        Args:
            query: 查询文本
            k: 召回上限（默认 config.memory_recall_k）
            types: 类型过滤（None=全部类型）
            min_importance: 重要性下限（默认 config.memory_min_importance）
            user_id: 命名空间过滤
            now: 时间基准（测试注入；默认当前时间）
        """
        store = None
        try:
            store = self._ensure_store()
            if store is None or not query.strip():
                return []

            top_k = k if k is not None else config.memory_recall_k
            floor = min_importance if min_importance is not None else config.memory_min_importance
            current_time = now if now is not None else time.time()

            query_vec = await self._embed_safe(query)
            candidates = await asyncio.to_thread(store.candidates, user_id=user_id, types=types)

            # 维度失配防护：换嵌入模型后存量向量与新查询全量不可比，
            # 若不拦截，相关性分量静默归零，召回会按重要性注入无关记忆。
            # 失配条目直接跳过（宁可少召回，不注垃圾），并告警提示重建索引。
            if query_vec:
                bad_dim_ids = {
                    c.id for c in candidates if c.embedding and len(c.embedding) != len(query_vec)
                }
                if bad_dim_ids:
                    first_bad = next(c for c in candidates if c.id in bad_dim_ids)
                    logger.warning(
                        f"embedding 维度不匹配: query={len(query_vec)} "
                        f"stored={len(first_bad.embedding or [])}，"
                        f"跳过 {len(bad_dim_ids)} 条存量向量（疑似更换嵌入模型，请执行 reindex）"
                    )
                    candidates = [c for c in candidates if c.id not in bad_dim_ids]

            scored: list[tuple[float, MemoryItem]] = []
            for item in candidates:
                if item.importance < floor:
                    continue
                score, _ = composite_score(
                    item,
                    query_vec or [],
                    self._weights(),
                    current_time,
                    config.memory_decay_lambda,
                )
                scored.append((score, item))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            hits = [item for score, item in scored[:top_k] if score > 0.0]
        except Exception as e:
            logger.warning(f"记忆召回失败（忽略）: {e}")
            return []

        if hits:
            hit_ids = [h.id for h in hits]
            try:
                task = self.write_queue.submit(
                    lambda: asyncio.to_thread(store.touch, hit_ids, current_time)
                )
                bg = asyncio.ensure_future(_swallow(task))
                self._bg.add(bg)
                bg.add_done_callback(self._bg.discard)
            except Exception as e:  # pragma: no cover - 触达统计尽力而为
                logger.debug(f"召回触达更新失败: {e}")

        logger.debug(f"记忆召回: query={query[:30]!r} 候选={len(candidates)} 命中={len(hits)}")
        return hits

    # ──────────────── 巩固 ────────────────

    async def consolidate(
        self,
        *,
        threshold: float | None = None,
        min_cluster: int = 2,
    ) -> dict[str, Any]:
        """情景记忆 → 语义记忆：贪心聚类合并

        对带向量的情景记忆按 created_at 升序扫描：每个未分配的记忆作为种子，
        吸收与其余弦相似度 ≥ threshold 的后续记忆形成簇；簇大小 ≥ min_cluster
        时生成一条语义记忆（内容为要点列表、向量为成员均值再归一化、重要性取
        成员最大值），成员软删除并回填 consolidated_into 引用。孤立情景保持原样。

        Returns:
            统计 dict：{"clusters", "members_consolidated", "semantic_ids"}
        """
        store = self._ensure_store()
        if store is None:
            return {"clusters": 0, "members_consolidated": 0, "semantic_ids": []}

        sim_threshold = threshold if threshold is not None else config.memory_consolidate_threshold

        try:
            episodics = [
                item
                for item in await asyncio.to_thread(
                    store.candidates, user_id=None, types=[MemoryType.EPISODIC]
                )
                if item.embedding
            ]
            episodics.sort(key=lambda it: it.created_at)

            assigned: set[str] = set()
            clusters: list[list[MemoryItem]] = []
            for i, seed in enumerate(episodics):
                if seed.id in assigned:
                    continue
                cluster = [seed]
                for other in episodics[i + 1 :]:
                    if other.id in assigned:
                        continue
                    assert seed.embedding is not None and other.embedding is not None
                    if cosine_similarity(seed.embedding, other.embedding) >= sim_threshold:
                        cluster.append(other)
                if len(cluster) >= min_cluster:
                    clusters.append(cluster)
                    assigned.update(m.id for m in cluster)

            semantic_ids: list[str] = []
            members_consolidated = 0
            for cluster in clusters:
                merged = self._merge_cluster(cluster)
                await self._submit_store("add", merged)
                await asyncio.to_thread(store.mark_consolidated, [m.id for m in cluster], merged.id)
                semantic_ids.append(merged.id)
                members_consolidated += len(cluster)
                logger.info(
                    f"记忆巩固: {len(cluster)} 条情景 → 语义记忆 {merged.id[:8]} "
                    f"({merged.content[:40]!r})"
                )
        except Exception as e:  # noqa: BLE001 - 门面级失败安全契约
            logger.warning(f"记忆巩固失败（忽略）: {e}")
            return {"clusters": 0, "members_consolidated": 0, "semantic_ids": []}

        return {
            "clusters": len(clusters),
            "members_consolidated": members_consolidated,
            "semantic_ids": semantic_ids,
        }

    def _merge_cluster(self, cluster: list[MemoryItem]) -> MemoryItem:
        """簇 → 单条语义记忆（确定性合并，不调 LLM）"""
        bullets = []
        for member in cluster:
            preview = member.content.replace("\n", " ")[:200]
            bullets.append(f"- {preview}")
        content = "[巩固经验] 基于 {} 条相似情景归纳：\n{}".format(len(cluster), "\n".join(bullets))

        assert cluster[0].embedding is not None
        mean_vec = _renormalize(_mean_vector([m.embedding or [] for m in cluster]))

        return MemoryItem(
            type=MemoryType.SEMANTIC,
            content=content,
            importance=max(m.importance for m in cluster),
            embedding=mean_vec,
            user_id=cluster[0].user_id,
            metadata={
                "consolidated_from": [m.id for m in cluster],
                "member_count": len(cluster),
                "consolidated_at": time.time(),
            },
            created_at=max(m.created_at for m in cluster),
        )

    # ──────────────── 管理接口 ────────────────

    async def forget_user(self, user_id: str) -> int:
        """软删除某用户全部记忆（API DELETE 语义），返回条数；失败安全返回 0"""
        store = self._ensure_store()
        if store is None:
            return 0
        try:
            count: int = await self._submit_store("soft_delete_by_user", user_id)
        except Exception as e:  # noqa: BLE001 - 门面级失败安全契约
            logger.warning(f"遗忘用户失败（忽略）: {e}")
            return 0
        logger.info(f"已遗忘用户 {user_id} 的 {count} 条记忆")
        return count

    async def list_items(
        self,
        *,
        user_id: str | None = None,
        types: list[MemoryType] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """列举记忆（API GET 语义）"""
        store = self._ensure_store()
        if store is None:
            return []
        result: list[MemoryItem] = await asyncio.to_thread(
            store.list_items,
            user_id=user_id,
            types=types,
            include_deleted=include_deleted,
            limit=limit,
        )
        return result

    async def stats(self) -> dict[str, Any]:
        """存储统计"""
        store = self._ensure_store()
        if store is None:
            return {"by_type": {}, "enabled": self.enabled}
        data: dict[str, Any] = await asyncio.to_thread(store.stats)
        data["enabled"] = self.enabled
        return data

    async def start(self) -> bool:
        """应用启动钩子：启动写队列 worker 并预热存储

        Returns:
            是否就绪（disabled 或建库失败返回 False，供健康注册判断）
        """
        if not self.enabled:
            logger.info("长期记忆功能未启用（memory_enabled=false）")
            return False
        self._stopped = False  # 支持同进程重启
        self.write_queue.start()
        if self._ensure_store() is None:
            return False
        assert self._store is not None
        stats = await asyncio.to_thread(self._store.stats)
        total = sum(v.get("active", v["total"]) for v in stats["by_type"].values())
        logger.info(f"长期记忆服务就绪（{total} 条存量记忆）")
        return True

    async def stop(self) -> None:
        """应用关闭钩子：阻断新调用 → 排空写队列 → 等在途后台任务 → 关闭存储"""
        self._stopped = True  # 先置位：杜绝 stop 期间新调用触发惰性重建
        await self.write_queue.stop()
        if self._bg:
            # 在途 fire-and-forget touch 等 1s 让其落库（尽力而为）
            await asyncio.wait(set(self._bg), timeout=1.0)
        if self._store is not None:
            await asyncio.to_thread(self._store.close)
            self._store = None

    # ──────────────── 内部 ────────────────

    async def _embed_safe(self, text: str) -> list[float] | None:
        """嵌入查询/文档；失败返回 None（降级为无向量参与）"""
        if self.embedder is None:
            return None
        try:
            embed_query_safe: Callable[[str], list[float] | None] = self.embedder.embed_query_safe
            result = await asyncio.to_thread(embed_query_safe, text)
            return result
        except Exception as e:
            logger.warning(f"嵌入失败（降级为无向量）: {e}")
            return None

    async def _submit_store(self, method: str, *args: Any) -> Any:
        """把同步 store 操作经写队列调度到线程池执行"""

        def factory() -> Any:
            store = self._store
            assert store is not None, "store 未初始化"
            return asyncio.to_thread(getattr(store, method), *args)

        return await self.write_queue.submit(factory)


async def _swallow(future: asyncio.Future[Any]) -> None:
    """吞掉后台任务的异常（仅记日志）——用于非关键路径的 fire-and-forget"""
    try:
        await future
    except Exception as e:  # noqa: BLE001
        logger.debug(f"后台记忆操作失败（忽略）: {e}")


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """逐维平均（输入假定同维、非空）"""
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i, x in enumerate(vec):
            sums[i] += x
    n = float(len(vectors))
    return [s / n for s in sums]


def _renormalize(vec: list[float]) -> list[float]:
    """归一化到单位长度（零向量原样返回）"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return vec
    return [x / norm for x in vec]


# 全局单例（惰性建库；lifespan 中 start()/stop()）
memory_service = MemoryService()
