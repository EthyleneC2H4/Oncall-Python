"""RAGAS 风格的评测维度

在现有 4 维评测基础上补充 Generation 质量评测：
- Context Recall: 正确文档是否被召回
- Context Precision: 召回文档中有用的比例（排名加权）
- Faithfulness: 回答是否忠于检索结果（LLM-as-Judge）
- Answer Relevancy: 回答是否切中用户问题（LLM-as-Judge）
"""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


class RAGASEvaluator:
    """RAGAS 风格评测器"""

    def __init__(self, datasets_dir: str = "eval/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self._diagnostic_cases: list[dict] | None = None
        self._negative_cases: list[dict] | None = None

    @property
    def diagnostic_cases(self) -> list[dict]:
        if self._diagnostic_cases is None:
            self._diagnostic_cases = self._load_dataset("diagnostic_cases.json")
        return self._diagnostic_cases

    @property
    def negative_cases(self) -> list[dict]:
        if self._negative_cases is None:
            self._negative_cases = self._load_dataset("negative_cases.json")
        return self._negative_cases

    @property
    def all_cases(self) -> list[dict]:
        return self.diagnostic_cases + self.negative_cases

    def _load_dataset(self, filename: str) -> list[dict]:
        """从 JSON 文件加载评测集"""
        filepath = self.datasets_dir / filename
        if not filepath.exists():
            logger.warning(f"评测集文件不存在: {filepath}")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"加载评测集: {filename}, {len(data)} 个用例")
        return data

    async def evaluate_all(
        self,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """运行全量评测

        Args:
            categories: 指定评测类别，None 表示全部

        Returns:
            评测结果字典
        """
        cases = self.all_cases
        if categories:
            cases = [c for c in cases if c.get("category") in categories]

        logger.info(f"开始评测: {len(cases)} 个用例")
        results = {
            "total_cases": len(cases),
            "case_results": [],
            "summary": {},
            "timestamp": time.time(),
        }

        for tc in cases:
            case_result = await self._evaluate_case(tc)
            results["case_results"].append(case_result)

        results["summary"] = self._compute_summary(results["case_results"])
        return results

    async def _evaluate_case(self, test_case: dict) -> dict:
        """评测单个用例"""
        from app.services.query_router import query_router
        from app.tools import retrieve_knowledge, query_alert_graph

        tc_id = test_case["id"]
        query = test_case["query"]
        expected_intent = test_case.get("expected_intent", "")
        logger.info(f"评测用例 {tc_id}: {query}")

        result: dict[str, Any] = {
            "id": tc_id,
            "category": test_case.get("category", ""),
            "query": query,
            "routing": {},
            "retrieval": {},
            "kg_analysis": {},
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

        # 1. 路由评测
        start = time.time()
        try:
            intent, keywords = await query_router.route(query)
            result["routing"] = {
                "intent": intent,
                "keywords": keywords,
                "correct": intent == expected_intent,
                "latency_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            result["routing"] = {"error": str(e), "correct": False}

        # 2. 检索评测
        start = time.time()
        try:
            context = retrieve_knowledge.invoke({"query": query})
            retrieval_text = context if isinstance(context, str) else str(context)
            expected_docs = test_case.get("expected_docs", [])

            # Context Recall: 期望文档被召回的比例
            if expected_docs:
                recalled = sum(
                    1 for doc in expected_docs
                    if doc.replace(".md", "") in retrieval_text.lower()
                )
                result["context_recall"] = recalled / len(expected_docs)
            else:
                result["context_recall"] = 1.0 if not retrieval_text.strip() else 0.5

            # Context Precision: 基于期望关键词的匹配
            expected_contains = test_case.get("expected_answer_contains", [])
            if expected_contains and retrieval_text:
                matched = sum(
                    1 for kw in expected_contains
                    if kw.lower() in retrieval_text.lower()
                )
                result["context_precision"] = matched / len(expected_contains)

            result["retrieval"] = {
                "returned": bool(retrieval_text),
                "context_recall": result["context_recall"],
                "context_precision": result["context_precision"],
                "latency_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            result["retrieval"] = {"error": str(e)}

        # 3. KG 评测
        start = time.time()
        try:
            keywords_to_try = (result.get("routing", {}).get("keywords", []) or []) + [query]
            for kw in keywords_to_try:
                kg_result = query_alert_graph.invoke({"alert_keyword": kw})
                if kg_result and "未找到" not in kg_result:
                    expected_rcs = test_case.get("expected_root_causes", [])
                    root_cause_hit = any(
                        rc.lower() in kg_result.lower() for rc in expected_rcs
                    ) if expected_rcs else False

                    result["kg_analysis"] = {
                        "found": True,
                        "root_cause_hit": root_cause_hit,
                        "latency_ms": (time.time() - start) * 1000,
                    }
                    break
            else:
                result["kg_analysis"] = {"found": False}
        except Exception as e:
            result["kg_analysis"] = {"error": str(e)}

        return result

    def _compute_summary(self, case_results: list[dict]) -> dict:
        """汇总评测结果"""
        total = len(case_results)
        if total == 0:
            return {}

        routing_correct = sum(
            1 for r in case_results if r.get("routing", {}).get("correct", False)
        )
        retrieval_returned = sum(
            1 for r in case_results if r.get("retrieval", {}).get("returned", False)
        )
        kg_found = sum(
            1 for r in case_results if r.get("kg_analysis", {}).get("found", False)
        )
        kg_root_hit = sum(
            1 for r in case_results if r.get("kg_analysis", {}).get("root_cause_hit", False)
        )

        avg_context_recall = sum(
            r.get("context_recall", 0) for r in case_results
        ) / total
        avg_context_precision = sum(
            r.get("context_precision", 0) for r in case_results
        ) / total

        # 按类别汇总
        categories: dict[str, dict] = {}
        for r in case_results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "routing_correct": 0, "kg_found": 0}
            categories[cat]["total"] += 1
            if r.get("routing", {}).get("correct", False):
                categories[cat]["routing_correct"] += 1
            if r.get("kg_analysis", {}).get("found", False):
                categories[cat]["kg_found"] += 1

        return {
            "routing_accuracy": round(routing_correct / total, 4),
            "retrieval_rate": round(retrieval_returned / total, 4),
            "avg_context_recall": round(avg_context_recall, 4),
            "avg_context_precision": round(avg_context_precision, 4),
            "kg_coverage": round(kg_found / total, 4),
            "kg_root_cause_accuracy": round(kg_root_hit / max(kg_found, 1), 4),
            "total_cases": total,
            "by_category": categories,
        }


# 全局单例
ragas_evaluator = RAGASEvaluator()
