"""打分纯函数的数学单测（不依赖任何基础设施）"""

import math

from app.services.memory.scoring import (
    SECONDS_PER_DAY,
    ScoreWeights,
    composite_score,
    cosine_similarity,
    recency_score,
)
from app.services.memory.types import MemoryItem, MemoryType


class TestCosineSimilarity:
    def test_identical_unit_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_unnormalized_same_direction(self):
        assert cosine_similarity([3.0, 0.0], [2.0, 0.0]) == 1.0

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_and_mismatched(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestRecencyScore:
    def test_fresh_is_one(self):
        assert recency_score(0.0, decay_lambda=0.05) == 1.0

    def test_one_day_half_life(self):
        """λ=ln2 时一天后得分恰为 0.5"""
        assert math.isclose(recency_score(SECONDS_PER_DAY, math.log(2)), 0.5)

    def test_zero_lambda_disables_decay(self):
        assert recency_score(365 * SECONDS_PER_DAY, 0.0) == 1.0

    def test_negative_age_clamped_to_zero(self):
        """未来时间戳（时钟偏差）不应产生 >1 的加分"""
        assert recency_score(-1000.0, 0.05) == 1.0


class TestCompositeScore:
    def _item(self, embedding=None, importance=0.5, age_seconds=0.0, now=1000.0):
        return MemoryItem(
            type=MemoryType.EPISODIC,
            content="x",
            importance=importance,
            embedding=embedding,
            created_at=now - age_seconds,
        )

    def test_perfect_match_all_components(self):
        now = 1000.0
        item = self._item(embedding=[1.0, 0.0], importance=1.0)
        score, breakdown = composite_score(
            item, [1.0, 0.0], ScoreWeights(0.6, 0.25, 0.15), now=now, decay_lambda=0.05
        )
        assert score == 1.0
        assert breakdown["relevance"] == 1.0
        assert breakdown["recency"] == 1.0

    def test_weights_respected(self):
        now = 1000.0
        item = self._item(embedding=None, importance=0.8)
        weights = ScoreWeights(0.6, 0.25, 0.15)
        score, _ = composite_score(item, [], weights, now=now, decay_lambda=0.05)
        # 无向量：只有重要性 + 新近度分量
        assert math.isclose(score, 0.25 * 0.8 + 0.15 * 1.0)

    def test_no_embedding_zero_relevance(self):
        item = self._item(embedding=None, importance=1.0)
        score, breakdown = composite_score(
            item, [1.0], ScoreWeights(0.6, 0.25, 0.15), now=1000.0, decay_lambda=0.05
        )
        assert breakdown["relevance"] == 0.0
        assert score == 0.4

    def test_negative_cosine_clamped(self):
        """反向向量的负余弦应钳为 0，不得拉低总分"""
        item = self._item(embedding=[-1.0, 0.0], importance=0.0)
        score, breakdown = composite_score(
            item, [1.0, 0.0], ScoreWeights(0.6, 0.25, 0.15), now=1000.0, decay_lambda=0.05
        )
        assert breakdown["relevance"] == 0.0

    def test_importance_clamped_above_one(self):
        item = self._item(embedding=None, importance=5.0)
        _, breakdown = composite_score(
            item, [], ScoreWeights(0.6, 0.25, 0.15), now=1000.0, decay_lambda=0.05
        )
        assert breakdown["importance"] == 1.0

    def test_age_in_breakdown(self):
        item = self._item(embedding=None, age_seconds=10 * SECONDS_PER_DAY)
        _, breakdown = composite_score(item, [], ScoreWeights(), now=1000.0, decay_lambda=0.1)
        assert math.isclose(breakdown["age_days"], 10.0)
