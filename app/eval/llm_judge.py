"""LLM-as-Judge 评测

使用 LLM 作为评委，对 RAG 系统的生成质量进行评分：
- Faithfulness: 回答中的每个断言是否可从 context 中推出
- Answer Relevancy: 回答是否切中用户问题
- Pairwise (P4): 两个回答对比裁决 → 胜率统计；两评委标注序列算 Cohen's κ
"""

import json
import re

from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory

FAITHFULNESS_PROMPT = """你是一个严格的事实核查评审员。请判断以下回答中的每个论断是否可以从给定的上下文中推导出来。

## 上下文（检索到的文档）
{context}

## 回答
{answer}

## 评分标准
- 5分：所有论断都有上下文支撑
- 4分：大部分论断有上下文支撑，少数为合理推理
- 3分：部分论断有支撑，部分为推测
- 2分：较多论断缺乏上下文支撑
- 1分：大部分论断与上下文无关或矛盾

请只返回一个 JSON：{{"score": N, "reason": "简要说明"}}"""


RELEVANCY_PROMPT = """你是一个回答质量评审员。请判断以下回答是否切中了用户的问题。

## 用户问题
{question}

## 回答
{answer}

## 评分标准
- 5分：直接、完整地回答了问题
- 4分：回答了问题的主要方面
- 3分：部分回答了问题
- 2分：回答偏离了问题重点
- 1分：完全没有回答问题

请只返回一个 JSON：{{"score": N, "reason": "简要说明"}}"""


class LLMJudge:
    """LLM-as-Judge 评测器"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.create_chat_model(
                streaming=False,
                model=config.rag_model,
                temperature=0,
            )
        return self._llm

    async def judge_faithfulness(self, context: str, answer: str) -> dict:
        """评测 Faithfulness: 回答是否忠于检索结果

        Args:
            context: 检索到的上下文
            answer: 模型生成的回答

        Returns:
            {"score": 1-5, "reason": "..."}
        """
        try:
            prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
            result = await self.llm.ainvoke(prompt)
            return self._parse_judge_result(result.content)
        except Exception as e:
            logger.error(f"Faithfulness 评测失败: {e}")
            return {"score": 0, "reason": f"评测失败: {e}"}

    async def judge_relevancy(self, question: str, answer: str) -> dict:
        """评测 Answer Relevancy: 回答是否切中问题

        Args:
            question: 用户问题
            answer: 模型生成的回答

        Returns:
            {"score": 1-5, "reason": "..."}
        """
        try:
            prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
            result = await self.llm.ainvoke(prompt)
            return self._parse_judge_result(result.content)
        except Exception as e:
            logger.error(f"Relevancy 评测失败: {e}")
            return {"score": 0, "reason": f"评测失败: {e}"}

    async def judge_full(self, question: str, context: str, answer: str) -> dict:
        """完整评测: Faithfulness + Relevancy

        Returns:
            {"faithfulness": {...}, "relevancy": {...}, "avg_score": float}
        """
        faithfulness = await self.judge_faithfulness(context, answer)
        relevancy = await self.judge_relevancy(question, answer)

        scores = [
            faithfulness.get("score", 0),
            relevancy.get("score", 0),
        ]
        valid_scores = [s for s in scores if s > 0]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "avg_score": round(avg, 2),
        }

    def _parse_judge_result(self, content: str) -> dict:
        """解析 LLM Judge 的 JSON 输出"""
        content = content.strip()
        # 尝试直接解析
        try:
            parsed: dict = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            pass

        # 提取 JSON
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            try:
                parsed_match: dict = json.loads(match.group())
                return parsed_match
            except json.JSONDecodeError:
                pass

        # 尝试提取数字
        score_match = re.search(r"(\d)", content)
        score = int(score_match.group(1)) if score_match else 0
        return {"score": score, "reason": content[:200]}


    # ──────────────── P4: Pairwise 对比评测 ────────────────

    async def judge_pairwise(
        self, question: str, answer_a: str, answer_b: str
    ) -> dict:
        """两个回答的对比裁决（位置偏差控制：先 A 后 B 单次裁决）

        Returns:
            {"winner": "A"|"B"|"tie", "reason": "..."}
        """
        prompt = PAIRWISE_PROMPT.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )
        try:
            result = await self.llm.ainvoke(prompt)
            parsed = _parse_pairwise(result.content)
            logger.info(f"Pairwise 裁决: winner={parsed['winner']}")
            return parsed
        except Exception as e:
            logger.error(f"Pairwise 评测失败: {e}")
            return {"winner": "tie", "reason": f"裁决失败: {e}"}


PAIRWISE_PROMPT = """你是严格的运维诊断质量评审员。对比同一个问题的两个回答，选出更好的一个。

## 用户问题
{question}

## 回答 A
{answer_a}

## 回答 B
{answer_b}

## 评审标准（按优先级）
1. 根因定位是否准确且有证据支撑
2. 处置建议是否可执行、风险是否提示
3. 是否包含编造信息

请只返回一个 JSON：{{"winner": "A" 或 "B" 或 "tie", "reason": "一句话理由"}}"""


def _parse_pairwise(content: str) -> dict:
    """解析 pairwise 裁决输出；解析失败保守判 tie"""
    content = (content or "").strip()
    try:
        parsed = json.loads(content)
        winner = str(parsed.get("winner", "tie")).upper()
        return {"winner": _normalize_winner(winner), "reason": str(parsed.get("reason", ""))}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            winner = str(parsed.get("winner", "tie")).upper()
            return {"winner": _normalize_winner(winner), "reason": str(parsed.get("reason", ""))}
        except json.JSONDecodeError:
            pass

    # 最后兜底：裸字母
    bare = re.search(r"\b([AB])\b", content.upper())
    if bare:
        return {"winner": bare.group(1), "reason": content[:120]}
    return {"winner": "tie", "reason": content[:120] or "无法解析裁决输出"}


def _normalize_winner(raw: str) -> str:
    if raw in ("A", "B"):
        return raw
    if raw in ("TIE", "DRAW", ""):
        return "tie"
    return "tie"


def pairwise_win_rate(judgements: list[dict]) -> dict:
    """聚合 pairwise 裁决序列 → 各方胜率（纯函数）

    Args:
        judgements: [{"winner": "A"|"B"|"tie"}, ...]

    Returns:
        {"a_wins", "b_wins", "ties", "total", "win_rate_a"}
    """
    a_wins = sum(1 for j in judgements if j.get("winner") == "A")
    b_wins = sum(1 for j in judgements if j.get("winner") == "B")
    ties = sum(1 for j in judgements if j.get("winner") == "tie")
    total = len(judgements)
    decisive = a_wins + b_wins
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": total,
        # 胜率只按分出胜负的对局计——平局稀释胜率没有意义
        "win_rate_a": round(a_wins / decisive, 4) if decisive else 0.0,
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """两评委标注序列的 Cohen's κ（纯函数）

    κ = (p_o − p_e) / (1 − p_e)，p_o 为观测一致率，p_e 为边缘分布期望一致率。
    κ=1 完全一致，κ=0 与随机猜测持平，κ<0 意见系统性相左。

    契约：长度不等或为空 → 抛 ValueError（调用方负责对齐）。
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"标注序列长度不等: {len(labels_a)} vs {len(labels_b)}")
    n = len(labels_a)
    if n == 0:
        raise ValueError("标注序列为空")

    p_o = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    categories = set(labels_a) | set(labels_b)
    p_e = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if p_e == 1.0:
        # 两评委各自只标一类：观测一致率必为 1，约定返回 1.0
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


# 全局单例
llm_judge = LLMJudge()
