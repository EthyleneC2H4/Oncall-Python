"""TTLCache 缓存模块单元测试

测试覆盖：CRUD、TTL 过期、LRU 淘汰、线程安全、统计。
与 app/core/cache.py 的真实 API 对齐：
- 构造签名 TTLCache(name, maxsize, ttl_seconds)
- get(key) 不支持 default 参数（未命中/过期一律返回 None）
- stats 为 property，hit_rate 为格式化百分比字符串
"""

import time

import pytest

from app.core.cache import TTLCache


class TestTTLCacheBasic:
    """基本 CRUD 操作"""

    def test_set_and_get(self, clean_cache):
        clean_cache.set("key1", "value1")
        assert clean_cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, clean_cache):
        assert clean_cache.get("nonexistent") is None

    def test_invalidate(self, clean_cache):
        clean_cache.set("key1", "value1")
        clean_cache.invalidate("key1")
        assert clean_cache.get("key1") is None

    def test_invalidate_nonexistent_is_noop(self, clean_cache):
        clean_cache.invalidate("nonexistent")  # 不应抛异常
        assert clean_cache.stats["size"] == 0

    def test_clear(self, clean_cache):
        for i in range(5):
            clean_cache.set(f"key{i}", f"value{i}")
        clean_cache.clear()
        assert clean_cache.get("key0") is None
        assert clean_cache.stats["size"] == 0


class TestTTLCacheTTL:
    """TTL 过期测试"""

    def test_expired_entry_returns_none(self):
        cache = TTLCache(name="ttl-test", maxsize=10, ttl_seconds=0.01)
        cache.set("key", "value")
        assert cache.get("key") == "value"
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_accessed_before_expiry_still_valid(self):
        cache = TTLCache(name="ttl-test", maxsize=10, ttl_seconds=0.05)
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") == "value"

    def test_no_ttl_never_expires(self):
        cache = TTLCache(name="ttl-test", maxsize=10, ttl_seconds=0)
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") == "value"


class TestTTLCacheLRU:
    """LRU 淘汰测试"""

    def test_max_size_eviction(self):
        cache = TTLCache(name="lru-test", maxsize=3, ttl_seconds=60)
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")
        # 最早的 key0 和 key1 应该被淘汰
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        assert cache.stats["size"] == 3

    def test_get_updates_lru_order(self):
        cache = TTLCache(name="lru-test", maxsize=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # 访问 a 提升到最新位置
        cache.get("a")
        # 插入 d，应该淘汰 b（a 被重新访问过）
        cache.set("d", 4)
        assert cache.get("a") == 1  # should survive
        assert cache.get("b") is None  # should be evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestTTLCacheStats:
    """统计测试"""

    def test_hit_and_miss_counts(self, clean_cache):
        clean_cache.set("key", "value")
        clean_cache.get("key")  # hit
        clean_cache.get("key")  # hit
        clean_cache.get("missing")  # miss

        stats = clean_cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "66.7%"
        assert stats["name"] == "test"
        assert stats["maxsize"] == 10

    def test_empty_cache_stats(self, clean_cache):
        stats = clean_cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == "0.0%"


class TestTTLCacheThreadSafety:
    """线程安全测试"""

    def test_concurrent_set_get(self):
        import threading

        # 容量需覆盖全部并发写入的 key（4 线程 × 50 个）
        cache = TTLCache(name="stress", maxsize=1000, ttl_seconds=60)

        def writer(start, count):
            for i in range(start, start + count):
                cache.set(f"key{i}", i)
                cache.get(f"key{i}")

        threads = [threading.Thread(target=writer, args=(i * 100, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证没有损坏（线程 3 的写入范围为 300-349）
        assert cache.get("key0") == 0
        assert cache.get("key349") == 349


@pytest.mark.parametrize(
    ("parts", "expect_same"),
    [
        (("a", "b"), True),
        (("a", "c"), False),
    ],
)
def test_make_cache_key_deterministic(parts, expect_same):
    """make_cache_key 对相同输入稳定、不同输入不同"""
    from app.core.cache import make_cache_key

    key_a = make_cache_key(*parts)
    key_b = make_cache_key("a", "b")
    assert (key_a == key_b) is expect_same
