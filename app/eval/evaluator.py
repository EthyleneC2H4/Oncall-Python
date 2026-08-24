"""AIOps 端到端评测框架

支持从外部 JSON 数据集加载评测用例。
评测维度：路由准确率、检索召回率、KG 覆盖率、根因命中率。
配合 ragas_evaluator.py 和 llm_judge.py 实现完整评测体系。
"""

import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.eval.dataset_registry import read_cases


class AIOpsEvaluator:
    """AIOps 端到端评测

    优先从 eval/datasets/ 加载用例，无文件时 fallback 到内置 5 个用例。
    """

    # 内置测试用例（向后兼容）
    _builtin_cases = [
        {
            "id": "TC001",
            "query": "CPU持续高于90%，伴随OOM日志",
            "expected_root_causes": ["内存泄漏", "MemoryLeak", "JVM参数配置不当"],
            "expected_actions": ["重启服务", "增大JVM堆内存", "导出内存快照分析"],
            "expected_cascade": ["FrequentGC", "OOMError", "SlowResponse"],
            "relevant_docs": ["cpu_high_usage.md", "memory_high_usage.md"],
        },
        {
            "id": "TC002",
            "query": "磁盘使用率超过90%，日志写入变慢",
            "expected_root_causes": ["日志文件过大", "LogFileTooLarge", "临时文件堆积"],
            "expected_actions": ["清理日志文件", "配置日志轮转", "磁盘扩容"],
            "expected_cascade": ["DiskIOHigh", "ServiceUnavailable"],
            "relevant_docs": ["disk_high_usage.md"],
        },
        {
            "id": "TC003",
            "query": "服务响应时间P99超过5秒",
            "expected_root_causes": ["数据库慢查询", "DBSlowQuery", "外部API调用超时"],
            "expected_actions": ["优化慢SQL", "降级非关键功能", "增加缓存"],
            "expected_cascade": ["HighErrorRate", "ServiceUnavailable"],
            "relevant_docs": ["slow_response.md"],
        },
        {
            "id": "TC004",
            "query": "服务不可用，健康检查持续失败",
            "expected_root_causes": ["应用崩溃", "AppCrash", "数据库连接失败"],
            "expected_actions": ["重启实例", "代码回滚", "切换备用服务"],
            "expected_cascade": [],
            "relevant_docs": ["service_unavailable.md"],
        },
        {
            "id": "TC005",
            "query": "内存使用率飙升至95%，GC频繁",
            "expected_root_causes": ["内存泄漏", "MemoryLeak", "缓存配置不当"],
            "expected_actions": ["导出内存快照分析", "重启实例", "修复内存泄漏"],
            "expected_cascade": ["OOMError", "HighCPUUsage", "ServiceUnavailable"],
            "relevant_docs": ["memory_high_usage.md"],
        },
    ]

    def __init__(self, datasets_dir: str = "eval/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self._test_cases: list[dict] | None = None

    @property
    def test_cases(self) -> list[dict]:
        """优先从文件加载，fallback 到内置用例"""
        if self._test_cases is None:
            self._test_cases = self._load_cases()
        return self._test_cases

    def _load_cases(self) -> list[dict]:
        """从 eval/datasets/ 加载用例（经 dataset_registry 兼容版本信封）"""
        cases = []
        diagnostic_file = self.datasets_dir / "diagnostic_cases.json"
        negative_file = self.datasets_dir / "negative_cases.json"

        for filepath in [diagnostic_file, negative_file]:
            if filepath.exists():
                try:
                    data = read_cases(filepath)
                    # 兼容旧格式：补充 relevant_docs 字段
                    for case in data:
                        if "relevant_docs" not in case:
                            case["relevant_docs"] = case.get("expected_docs", [])
                    cases.extend(data)
                    logger.info(f"加载评测集: {filepath.name}, {len(data)} 个用例")
                except Exception as e:
                    logger.warning(f"加载评测集失败 {filepath}: {e}")

        if not cases:
            logger.info("未找到外部评测集，使用内置 5 个用例")
            return list(self._builtin_cases)

        logger.info(f"评测集总计: {len(cases)} 个用例")
        return cases

    async def evaluate_all(self) -> dict[str, Any]:
        """运行所有测试用例，返回评测结果"""
        results: dict[str, Any] = {
            "total_cases": len(self.test_cases),
            "case_results": [],
            "summary": {},
        }

        case_results: list[Any] = results["case_results"]
        for tc in self.test_cases:
            case_results.append(await self._evaluate_case(tc))

        # 汇总
        results["summary"] = self._summarize(case_results)
        return results

    async def _evaluate_case(self, test_case: dict) -> dict:
        """评测单个测试用例"""
        from app.services.query_router import query_router
        from app.tools import query_alert_graph, retrieve_knowledge

        tc_id = test_case["id"]
        query = test_case["query"]
        logger.info(f"评测用例 {tc_id}: {query}")

        result = {
            "id": tc_id,
            "query": query,
            "routing": {},
            "retrieval": {},
            "kg_analysis": {},
            "latency": {},
        }

        # 1. 评测路由
        start = time.time()
        intent, keywords = await query_router.route(query)
        result["routing"] = {
            "intent": intent,
            "keywords": keywords,
            "correct": intent == "DIAGNOSTIC",
            "latency_ms": (time.time() - start) * 1000,
        }

        # 2. 评测检索
        start = time.time()
        try:
            context = retrieve_knowledge.invoke({"query": query})
            retrieval_text = context if isinstance(context, str) else str(context)
            # 检查是否召回了相关文档
            relevant_found = any(
                doc_name.replace(".md", "") in retrieval_text.lower()
                for doc_name in test_case.get("relevant_docs", [])
            )
            result["retrieval"] = {
                "returned": bool(retrieval_text),
                "relevant_found": relevant_found,
                "latency_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            result["retrieval"] = {"error": str(e)}

        # 3. 评测知识图谱
        start = time.time()
        try:
            for kw in keywords + [query]:
                kg_result = query_alert_graph.invoke({"alert_keyword": kw})
                if kg_result and "未找到" not in kg_result:
                    # 检查根因命中
                    root_cause_hit = any(
                        rc.lower() in kg_result.lower()
                        for rc in test_case.get("expected_root_causes", [])
                    )
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

    def _summarize(self, case_results: list) -> dict:
        """汇总评测结果"""
        total = len(case_results)
        if total == 0:
            return {}

        routing_correct = sum(1 for r in case_results if r.get("routing", {}).get("correct", False))
        retrieval_relevant = sum(
            1 for r in case_results if r.get("retrieval", {}).get("relevant_found", False)
        )
        kg_found = sum(1 for r in case_results if r.get("kg_analysis", {}).get("found", False))
        kg_root_hit = sum(
            1 for r in case_results if r.get("kg_analysis", {}).get("root_cause_hit", False)
        )

        return {
            "routing_accuracy": routing_correct / total,
            "retrieval_recall": retrieval_relevant / total,
            "kg_coverage": kg_found / total,
            "kg_root_cause_accuracy": kg_root_hit / total if kg_found > 0 else 0,
        }
