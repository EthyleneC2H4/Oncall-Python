"""通用 TTL-LRU 缓存

线程安全的内存缓存，支持 maxsize + TTL 淘汰。
用于 Embedding 缓存、LLM 响应缓存、检索结果缓存等。
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """线程安全的 LRU + TTL 缓存"""

    def __init__(self, name: str, maxsize: int = 1000, ttl_seconds: float = 3600.0):
        self.name = name
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, ts = entry
            if self._ttl > 0 and time.time() - ts > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(self._hits + self._misses, 1):.1%}",
        }


def make_cache_key(*parts: str) -> str:
    """从多个字符串生成缓存 key (SHA256)"""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------- 预定义缓存实例 ----------

# Embedding 向量缓存（确定性输出，不设 TTL）
embedding_cache = TTLCache("embedding", maxsize=5000, ttl_seconds=0)

# LLM 响应缓存（10 分钟 TTL）
llm_response_cache = TTLCache("llm_response", maxsize=200, ttl_seconds=600)

# 检索结果缓存（5 分钟 TTL）
retrieval_cache = TTLCache("retrieval", maxsize=500, ttl_seconds=300)
