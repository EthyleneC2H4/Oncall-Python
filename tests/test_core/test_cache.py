"""TTLCache 缓存模块单元测试

测试覆盖：CRUD、TTL 过期、LRU 淘汰、线程安全、统计。
"""

import time

import pytest


class TestTTLCacheBasic:
    """基本 CRUD 操作"""

    def test_set_and_get(self, clean_cache):
        clean_cache.set("key1", "value1")
        assert clean_cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, clean_cache):
        assert clean_cache.get("nonexistent") is None

    def test_get_missing_returns_default(self, clean_cache):
        assert clean_cache.get("nonexistent", "default") == "default"

    def test_invalidate(self, clean_cache):
        clean_cache.set("key1", "value1")
        clean_cache.invalidate("key1")
        assert clean_cache.get("key1") is None

    def test_clear(self, clean_cache):
        for i in range(5):
            clean_cache.set(f"key{i}", f"value{i}")
        clean_cache.clear()
        assert clean_cache.get("key0") is None


class TestTTLCacheTTL:
    """TTL 过期测试"""

    def test_expired_entry_returns_none(self, clean_cache):
        cache = clean_cache
        cache.ttl = 0.01  # 10ms TTL
        cache.set("key", "value")
        assert cache.get("key") == "value"
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_accessed_before_expiry_still_valid(self, clean_cache):
        cache = clean_cache
        cache.ttl = 0.05
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") == "value"

    def test_no_ttl_never_expires(self, clean_cache):
        cache = clean_cache
        cache.ttl = 0  # no TTL
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") == "value"


class TestTTLCacheLRU:
    """LRU 淘汰测试"""

    def test_max_size_eviction(self, clean_cache):
        cache = clean_cache
        cache.max_size = 3
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")
        # 最早的 key0 和 key1 应该被淘汰
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    def test_get_updates_lru_order(self, clean_cache):
        cache = clean_cache
        cache.max_size = 3
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # 访问 a 提升到最前面
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
        assert stats["hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_empty_cache_stats(self, clean_cache):
        stats = clean_cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0


class TestTTLCacheThreadSafety:
    """线程安全测试"""

    def test_concurrent_set_get(self, clean_cache):
        import threading

        def writer(start, count):
            for i in range(start, start + count):
                clean_cache.set(f"key{i}", i)
                clean_cache.get(f"key{i}")

        threads = [
            threading.Thread(target=writer, args=(i * 100, 50))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证没有损坏
        assert clean_cache.get("key0") == 0
        assert clean_cache.get("key350") == 350
