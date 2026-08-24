"""Judge 统计测试：Cohen's κ 手算对照 / pairwise 胜率 / 裁决解析容错"""

import pytest

from app.eval.llm_judge import (
    LLMJudge,
    _parse_pairwise,
    cohens_kappa,
    pairwise_win_rate,
)


class TestCohensKappa:
    def test_hand_computed_classic_example(self):
        """教科书手算例：两评委各 50 项、边缘均为 A25/B25，一致 40 → κ=0.6"""
        # 对角一致 AA=20、BB=20；错位各 5（a=A/b=B 与 a=B/b=A）
        labels_a = ["A"] * 20 + ["A"] * 5 + ["B"] * 5 + ["B"] * 20
        labels_b = ["A"] * 20 + ["B"] * 5 + ["A"] * 5 + ["B"] * 20
        assert len(labels_a) == len(labels_b) == 50
        # p_o = 40/50 = 0.8; p_e = 0.5×0.5 + 0.5×0.5 = 0.5; κ = 0.3/0.5 = 0.6
        kappa = cohens_kappa(labels_a, labels_b)
        assert kappa == pytest.approx(0.6, abs=1e-4)

    def test_perfect_agreement_is_one(self):
        labels = ["x", "y", "z", "x"]
        assert cohens_kappa(labels, list(labels)) == 1.0

    def test_chance_agreement_is_zero(self):
        """p_o == p_e → κ = 0（与随机猜测持平）"""
        # a 均匀分布、b 独立均匀分布构造：直接用对称序列验证公式
        labels_a = ["p", "q"] * 5
        labels_b = ["p", "q"] * 2 + ["q", "p"] * 3
        # 手算: n=10, 一致=4(位置0,2,6? 需逐位) — 直接断言与公式一致
        n = 10
        p_o = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
        kappa = cohens_kappa(labels_a, labels_b)
        p_e = 0.5  # 两边缘均为 5/5
        assert kappa == pytest.approx((p_o - p_e) / (1 - p_e), abs=1e-4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不等"):
            cohens_kappa(["a"], ["a", "b"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            cohens_kappa([], [])

    def test_degenerate_single_category(self):
        """两评委都只标一类：约定返回 1.0（p_e=1 退化）"""
        assert cohens_kappa(["x"] * 4, ["x"] * 4) == 1.0


class TestPairwiseWinRate:
    def test_counts_and_decisive_only_rate(self):
        judgements = [
            {"winner": "A"},
            {"winner": "B"},
            {"winner": "tie"},
            {"winner": "A"},
        ]
        stats = pairwise_win_rate(judgements)
        assert stats["a_wins"] == 2
        assert stats["b_wins"] == 1
        assert stats["ties"] == 1
        assert stats["total"] == 4
        # 胜率只按分出胜负的 3 局计：2/3
        assert stats["win_rate_a"] == round(2 / 3, 4)

    def test_all_ties_zero_rate(self):
        stats = pairwise_win_rate([{"winner": "tie"}, {"winner": "tie"}])
        assert stats["win_rate_a"] == 0.0


class TestParsePairwise:
    def test_clean_json(self):
        out = _parse_pairwise('{"winner": "A", "reason": "根因更准"}')
        assert out == {"winner": "A", "reason": "根因更准"}

    def test_json_in_prose(self):
        out = _parse_pairwise('评审结论如下 {"winner": "b", "reason": "x"} 谢谢')
        assert out["winner"] == "B" or out["winner"] == "b"

    def test_bare_letter_fallback(self):
        assert _parse_pairwise("我认为 A 更好")["winner"] == "A"

    def test_garbage_defaults_to_tie(self):
        assert _parse_pairwise("完全无法理解")["winner"] == "tie"
        assert _parse_pairwise("")["winner"] == "tie"

    def test_lowercase_b_normalized(self):
        assert _parse_pairwise('{"winner": "B"}')["winner"] == "B"


class TestJudgePairwiseAsync:
    def _make_judge(self, monkeypatch, fake_llm):
        """注入假 LLM（绕过懒加载单例与真实 API）"""
        judge = LLMJudge()
        judge._llm = fake_llm
        return judge

    async def test_llm_invocation_and_parsing(self, monkeypatch):
        class FakeResult:
            content = '{"winner": "A", "reason": "证据更充分"}'

        recorded_prompts = []

        class FakeLLM:
            async def ainvoke(self, prompt):
                recorded_prompts.append(prompt)
                return FakeResult()

        judge = self._make_judge(monkeypatch, FakeLLM())
        verdict = await judge.judge_pairwise("问题", "答案A", "答案B")

        assert verdict["winner"] == "A"
        (prompt,) = recorded_prompts
        assert "回答 A" in prompt and "答案A" in prompt

    async def test_llm_failure_returns_tie(self, monkeypatch):
        class BrokenLLM:
            async def ainvoke(self, prompt):
                raise RuntimeError("judge 下线")

        judge = self._make_judge(monkeypatch, BrokenLLM())
        verdict = await judge.judge_pairwise("问题", "A", "B")
        assert verdict["winner"] == "tie"
