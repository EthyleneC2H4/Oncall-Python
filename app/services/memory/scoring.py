"""长期记忆 - 打分纯函数

召回打分公式（权重入配置，默认 0.6 / 0.25 / 0.15）：

    score = w_relevance * cosine(query, item)
          + w_importance * item.importance
          + w_recency   * exp(-λ * age_days)

所有函数均为纯函数：时间由调用方注入（now），不读时钟、不碰 IO，
保证可做精确的数学单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.memory.types import MemoryItem

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class ScoreWeights:
    """打分权重（和不必为 1，但推荐为 1 以保持分数落在 [0,1]）"""

    relevance: float = 0.6
    importance: float = 0.25
    recency: float = 0.15


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量为零向量/空向量时返回 0.0

    输入约定为已归一化向量（BGE normalize_embeddings=True），
    此时等价于点积；这里仍按完整公式实现以容忍未归一化输入。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def recency_score(age_seconds: float, decay_lambda: float) -> float:
    """新近度得分：exp(-λ · 天数)

    Args:
        age_seconds: 距今秒数（负数视为 0 —— 未来时间不加分）
        decay_lambda: 每日衰减率 λ；λ≤0 时恒为 1（关闭衰减，负值视为配置笔误）
    """
    if decay_lambda <= 0.0:
        return 1.0
    age_days = max(age_seconds, 0.0) / SECONDS_PER_DAY
    return math.exp(-decay_lambda * age_days)


def composite_score(
    item: MemoryItem,
    query_embedding: list[float],
    weights: ScoreWeights,
    now: float,
    decay_lambda: float,
) -> tuple[float, dict[str, float]]:
    """综合打分：相关性 + 重要性 + 新近度加权求和

    Returns:
        (score, breakdown)：breakdown 含各分量与 age_days，便于调试与测试断言。
        无向量的记忆条目相关性记 0（仍可凭重要性/新近度参与排序）。
    """
    if item.embedding is not None and query_embedding:
        relevance = cosine_similarity(query_embedding, item.embedding)
        # 余弦可为负（罕见），钳到 [0,1] 避免负分干扰排序语义
        relevance = max(0.0, min(relevance, 1.0))
    else:
        relevance = 0.0

    age_seconds = now - item.created_at
    recency = recency_score(age_seconds, decay_lambda)
    importance = max(0.0, min(item.importance, 1.0))

    score = (
        weights.relevance * relevance
        + weights.importance * importance
        + weights.recency * recency
    )
    breakdown = {
        "relevance": round(relevance, 6),
        "importance": importance,
        "recency": round(recency, 6),
        "age_days": round(max(age_seconds, 0.0) / SECONDS_PER_DAY, 6),
        "score": round(score, 6),
    }
    return score, breakdown
