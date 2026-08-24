"""长期记忆服务包

四类记忆（working/episodic/semantic/procedural）+ sqlite 持久化 + 单 worker
写队列 + 纯函数打分 + 向量召回 + 情景→语义巩固。

对外入口：
- memory_service: 全局服务单例（惰性建库，lifespan 中 start/stop）
- MemoryItem / MemoryType: 数据类型
- MemoryStore / WriteQueue / ScoreWeights: 可独立注入的组件
"""

from app.services.memory.queue import WriteQueue
from app.services.memory.scoring import (
    ScoreWeights,
    composite_score,
    cosine_similarity,
    recency_score,
)
from app.services.memory.service import MemoryService, memory_service
from app.services.memory.store import MemoryStore
from app.services.memory.types import MemoryItem, MemoryType

__all__ = [
    "MemoryItem",
    "MemoryService",
    "MemoryStore",
    "MemoryType",
    "ScoreWeights",
    "WriteQueue",
    "composite_score",
    "cosine_similarity",
    "memory_service",
    "recency_score",
]
