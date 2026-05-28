"""LLM-as-Judge 评测

使用 LLM 作为评委，对 RAG 系统的生成质量进行评分：
- Faithfulness: 回答中的每个断言是否可从 context 中推出
- Answer Relevancy: 回答是否切中用户问题
"""

from loguru import logger
from langchain_qwq import ChatQwen

from app.config import config


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
            self._llm = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
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
        import json
        import re

        content = content.strip()
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 提取 JSON
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 尝试提取数字
        score_match = re.search(r'(\d)', content)
        score = int(score_match.group(1)) if score_match else 0
        return {"score": score, "reason": content[:200]}


# 全局单例
llm_judge = LLMJudge()
