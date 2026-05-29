"""RAGAS 框架评测模块

使用 RAGAS (Retrieval Augmented Generation Assessment) 框架进行标准化 RAG 评测。

支持两种评测模式：
1. 端到端评测 (e2e): question → retrieve → generate → RAGAS evaluate
   - Faithfulness: 回答中的断言是否可从上下文推出
   - Answer Relevancy: 回答是否切中用户问题
   - Context Recall: 检索上下文是否覆盖参考答案
   - Context Precision: 检索上下文中相关内容的排名质量

2. 组件评测 (component): 路由 + 检索 + KG 维度的快速评测（不调用 RAGAS）
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Any

from loguru import logger


class RAGASEvaluator:
    """RAGAS 框架评测器

    支持 e2e（端到端 RAGAS 评测）和 component（组件级快速评测）两种模式。
    """

    def __init__(self, datasets_dir: str = "eval/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self._diagnostic_cases: list[dict] | None = None
        self._negative_cases: list[dict] | None = None
        self._evaluator_llm = None
        self._evaluator_embeddings = None

    # ──────────────── 数据集加载 ────────────────

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
        filepath = self.datasets_dir / filename
        if not filepath.exists():
            logger.warning(f"评测集文件不存在: {filepath}")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"加载评测集: {filename}, {len(data)} 个用例")
        return data

    # ──────────────── RAGAS LLM / Embeddings ────────────────

    def _get_evaluator_llm(self):
        """获取 RAGAS 兼容的 LLM（延迟初始化）"""
        if self._evaluator_llm is None:
            from ragas.llms import LangchainLLMWrapper
            from langchain_qwq import ChatQwen
            from app.config import config

            llm = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=0,
            )
            self._evaluator_llm = LangchainLLMWrapper(llm)
        return self._evaluator_llm

    def _get_evaluator_embeddings(self):
        """获取 RAGAS 兼容的 Embeddings（延迟初始化）"""
        if self._evaluator_embeddings is None:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from app.services.vector_embedding_service import vector_embedding_service

            self._evaluator_embeddings = LangchainEmbeddingsWrapper(
                vector_embedding_service
            )
        return self._evaluator_embeddings

    # ──────────────── 上下文收集（Retrieve + Generate）────────────────

    async def _retrieve_context(self, query: str) -> list[str]:
        """检索上下文"""
        from app.tools import retrieve_knowledge

        try:
            context = await asyncio.to_thread(
                retrieve_knowledge.invoke, {"query": query}
            )
            text = context if isinstance(context, str) else str(context)
            if text.strip():
                return [text]
            return []
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

    async def _generate_answer(self, query: str, contexts: list[str]) -> str:
        """基于检索上下文生成回答"""
        from langchain_qwq import ChatQwen
        from app.config import config

        context_text = "\n\n".join(contexts) if contexts else "未检索到相关文档。"
        prompt = (
            "你是一个智能运维助手，请基于以下检索到的上下文回答用户的问题。\n"
            "如果上下文不包含相关信息，请基于运维知识给出建议。\n\n"
            f"## 检索上下文\n{context_text}\n\n"
            f"## 用户问题\n{query}\n\n"
            "## 回答\n"
        )
        try:
            llm = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=0,
            )
            result = await llm.ainvoke(prompt)
            return result.content
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"生成失败: {e}"

    async def _collect_sample(self, test_case: dict) -> dict:
        """收集单个评测样本：retrieve → generate → 组装"""
        query = test_case["query"]
        reference = test_case.get("reference", "")

        # 检索
        start = time.time()
        contexts = await self._retrieve_context(query)
        retrieval_latency = (time.time() - start) * 1000

        # 生成
        start = time.time()
        answer = await self._generate_answer(query, contexts)
        generation_latency = (time.time() - start) * 1000

        return {
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": generation_latency,
        }

    # ──────────────── 端到端 RAGAS 评测 ────────────────

    async def evaluate_e2e(
        self,
        categories: list[str] | None = None,
        metrics: list[str] | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        """端到端 RAGAS 评测

        Args:
            categories: 过滤类别 (easy/medium/hard/edge_case/chitchat/knowledge)
            metrics: 选择指标 (faithfulness/answer_relevancy/context_recall/context_precision)
                     None 表示全部
            max_cases: 最大评测用例数（用于限制耗时和成本）

        Returns:
            包含 RAGAS 评分、用例详情和汇总的字典
        """
        from ragas import evaluate as ragas_evaluate
        from ragas import EvaluationDataset, SingleTurnSample

        # 筛选用例
        cases = self.all_cases
        if categories:
            cases = [c for c in cases if c.get("category") in categories]
        if max_cases:
            cases = cases[:max_cases]

        logger.info(f"RAGAS E2E 评测开始: {len(cases)} 个用例")
        start_time = time.time()

        # 收集样本（retrieve + generate）
        samples = []
        sample_meta = []
        for tc in cases:
            logger.info(f"收集样本 {tc['id']}: {tc['query'][:30]}...")
            sample_data = await self._collect_sample(tc)
            sample_meta.append({
                "id": tc["id"],
                "category": tc.get("category", ""),
                "query": tc["query"],
                "response": sample_data["response"],
                "retrieved_contexts": sample_data["retrieved_contexts"],
                "reference": sample_data["reference"],
                "retrieval_latency_ms": sample_data["retrieval_latency_ms"],
                "generation_latency_ms": sample_data["generation_latency_ms"],
            })

            sample = SingleTurnSample(
                user_input=sample_data["user_input"],
                retrieved_contexts=sample_data["retrieved_contexts"] or [""],
                response=sample_data["response"],
                reference=sample_data["reference"],
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)

        # 构建指标列表
        ragas_metrics = self._build_metrics(metrics)
        logger.info(f"使用 RAGAS 指标: {[type(m).__name__ for m in ragas_metrics]}")

        # 运行 RAGAS 评测
        try:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=ragas_metrics,
                llm=self._get_evaluator_llm(),
                embeddings=self._get_evaluator_embeddings(),
            )

            # 解析结果
            scores_df = result.to_pandas()
            per_case_scores = scores_df.to_dict(orient="records")

            # 合并元信息和 RAGAS 评分
            case_results = []
            for i, meta in enumerate(sample_meta):
                case_result = {**meta}
                if i < len(per_case_scores):
                    case_result["ragas_scores"] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in per_case_scores[i].items()
                        if k not in ("user_input", "retrieved_contexts", "response", "reference")
                    }
                case_results.append(case_result)

            # 汇总
            summary = self._compute_ragas_summary(case_results, scores_df)
            total_time = time.time() - start_time

            return {
                "mode": "e2e",
                "total_cases": len(cases),
                "case_results": case_results,
                "summary": summary,
                "total_time_seconds": round(total_time, 2),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"RAGAS 评测失败: {e}")
            return {
                "mode": "e2e",
                "total_cases": len(cases),
                "error": str(e),
                "case_results": sample_meta,
                "timestamp": time.time(),
            }

    def _build_metrics(self, metric_names: list[str] | None = None):
        """构建 RAGAS 指标实例"""
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextRecall,
            LLMContextPrecisionWithReference,
        )

        available = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_recall": LLMContextRecall,
            "context_precision": LLMContextPrecisionWithReference,
        }

        if metric_names is None:
            metric_names = list(available.keys())

        metrics = []
        for name in metric_names:
            if name in available:
                metrics.append(available[name]())
            else:
                logger.warning(f"未知的 RAGAS 指标: {name}")

        if not metrics:
            # fallback 全部
            metrics = [cls() for cls in available.values()]

        return metrics

    def _compute_ragas_summary(self, case_results: list[dict], scores_df) -> dict:
        """计算 RAGAS 评分汇总"""
        summary: dict[str, Any] = {}

        # 全局平均分
        metric_columns = [
            c for c in scores_df.columns
            if c not in ("user_input", "retrieved_contexts", "response", "reference")
        ]
        for col in metric_columns:
            values = scores_df[col].dropna()
            if len(values) > 0:
                summary[f"avg_{col}"] = round(values.mean(), 4)
                summary[f"min_{col}"] = round(values.min(), 4)
                summary[f"max_{col}"] = round(values.max(), 4)

        # 按类别汇总
        by_category: dict[str, dict] = {}
        for cr in case_results:
            cat = cr.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"total": 0, "scores": {col: [] for col in metric_columns}}
            by_category[cat]["total"] += 1
            ragas_scores = cr.get("ragas_scores", {})
            for col in metric_columns:
                val = ragas_scores.get(col)
                if val is not None and isinstance(val, (int, float)):
                    by_category[cat]["scores"][col].append(val)

        for cat, data in by_category.items():
            cat_summary = {"total": data["total"]}
            for col, values in data["scores"].items():
                if values:
                    cat_summary[f"avg_{col}"] = round(sum(values) / len(values), 4)
            by_category[cat] = cat_summary

        summary["by_category"] = by_category
        return summary

    # ──────────────── 组件级快速评测 ────────────────

    async def evaluate_component(
        self,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """组件级评测（快速，不调用 RAGAS）

        评测维度：路由准确率、检索命中率、Context Recall/Precision、KG 覆盖率
        """
        cases = self.all_cases
        if categories:
            cases = [c for c in cases if c.get("category") in categories]

        logger.info(f"组件评测开始: {len(cases)} 个用例")
        start_time = time.time()

        results = []
        for tc in cases:
            result = await self._evaluate_component_case(tc)
            results.append(result)

        summary = self._compute_component_summary(results)
        total_time = time.time() - start_time

        return {
            "mode": "component",
            "total_cases": len(cases),
            "case_results": results,
            "summary": summary,
            "total_time_seconds": round(total_time, 2),
            "timestamp": time.time(),
        }

    async def _evaluate_component_case(self, test_case: dict) -> dict:
        """组件级评测单个用例"""
        from app.services.query_router import query_router
        from app.tools import retrieve_knowledge, query_alert_graph

        tc_id = test_case["id"]
        query = test_case["query"]
        expected_intent = test_case.get("expected_intent", "")
        logger.info(f"评测用例 {tc_id}: {query[:30]}...")

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
                "latency_ms": round((time.time() - start) * 1000, 2),
            }
        except Exception as e:
            result["routing"] = {"error": str(e), "correct": False}

        # 2. 检索评测
        start = time.time()
        try:
            context = retrieve_knowledge.invoke({"query": query})
            retrieval_text = context if isinstance(context, str) else str(context)
            expected_docs = test_case.get("expected_docs", [])

            # Context Recall
            if expected_docs:
                recalled = sum(
                    1 for doc in expected_docs
                    if doc.replace(".md", "") in retrieval_text.lower()
                )
                result["context_recall"] = round(recalled / len(expected_docs), 4)
            else:
                result["context_recall"] = 1.0 if not retrieval_text.strip() else 0.5

            # Context Precision
            expected_contains = test_case.get("expected_answer_contains", [])
            if expected_contains and retrieval_text:
                matched = sum(
                    1 for kw in expected_contains
                    if kw.lower() in retrieval_text.lower()
                )
                result["context_precision"] = round(matched / len(expected_contains), 4)

            result["retrieval"] = {
                "returned": bool(retrieval_text),
                "context_recall": result["context_recall"],
                "context_precision": result["context_precision"],
                "latency_ms": round((time.time() - start) * 1000, 2),
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
                        "latency_ms": round((time.time() - start) * 1000, 2),
                    }
                    break
            else:
                result["kg_analysis"] = {"found": False}
        except Exception as e:
            result["kg_analysis"] = {"error": str(e)}

        return result

    def _compute_component_summary(self, case_results: list[dict]) -> dict:
        """组件评测汇总"""
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

    # ──────────────── 统一入口 ────────────────

    async def evaluate_all(
        self,
        mode: str = "component",
        categories: list[str] | None = None,
        metrics: list[str] | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        """统一评测入口

        Args:
            mode: "e2e"（端到端 RAGAS）或 "component"（组件级快速评测）
            categories: 过滤类别
            metrics: RAGAS 指标选择（仅 e2e 模式有效）
            max_cases: 最大用例数（仅 e2e 模式有效）
        """
        if mode == "e2e":
            return await self.evaluate_e2e(
                categories=categories,
                metrics=metrics,
                max_cases=max_cases,
            )
        else:
            return await self.evaluate_component(categories=categories)

    # ──────────────── 结果持久化 ────────────────

    def save_results(self, results: dict, filename: str | None = None):
        """保存评测结果到 JSON 文件"""
        if filename is None:
            ts = int(time.time())
            mode = results.get("mode", "unknown")
            filename = f"eval_results_{mode}_{ts}.json"

        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"评测结果已保存: {output_path}")
        return str(output_path)


# 全局单例
ragas_evaluator = RAGASEvaluator()
