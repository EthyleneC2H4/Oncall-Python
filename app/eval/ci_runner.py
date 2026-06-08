"""CI 评测管道 — 门禁式 Agent 质量保障

整合 RAGAS + DeepEval + LLM Judge 三维评测，提供 CI/CD 可集成的评测管道。

功能：
1. 回归评测 (regression): 跑全部 55 个用例，对比上个版本的评分趋势
2. 门禁评测 (gating): 只跑关键用例，pass/fail 判定
3. 冒烟评测 (smoke): 跑 5 个核心用例，快速验证
4. 报告生成: 输出 Markdown + JSON 评测报告

使用方式：
    # 完整回归
    python -m app.eval.ci_runner --mode regression

    # CI 门禁
    python -m app.eval.ci_runner --mode gating --threshold 0.6

    # 冒烟
    python -m app.eval.ci_runner --mode smoke
"""

import json
import time
import argparse
from pathlib import Path
from typing import Any

from loguru import logger


# CI 门禁阈值配置
CI_THRESHOLDS = {
    "regression": {
        "routing_accuracy": 0.70,
        "avg_context_recall": 0.50,
        "avg_context_precision": 0.40,
        "kg_coverage": 0.30,
        "avg_faithfulness": 0.60,
        "avg_answer_relevancy": 0.60,
    },
    "gating": {
        "routing_accuracy": 0.60,
        "avg_context_recall": 0.40,
        "compound_score": 0.60,
    },
    "smoke": {
        "routing_accuracy": 0.50,
    },
}


class CIEvalRunner:
    """CI 评测管道运行器"""

    def __init__(self, output_dir: str = "eval/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────── 评测模式 ────────────────

    async def run_regression(self) -> dict[str, Any]:
        """回归评测：全部 55 用例，含 RAGAS 指标"""
        logger.info("=" * 60)
        logger.info("CI 回归评测开始")
        logger.info("=" * 60)

        from app.eval.ragas_evaluator import ragas_evaluator

        # 组件评测（快速）
        component_results = await ragas_evaluator.evaluate_component()
        summary = component_results.get("summary", {})

        # 检查门禁
        thresholds = CI_THRESHOLDS["regression"]
        gate_result = self._check_gate(summary, thresholds)

        report = {
            "mode": "regression",
            "timestamp": time.time(),
            "total_cases": component_results.get("total_cases", 0),
            "summary": summary,
            "gate_check": gate_result,
            "passed": gate_result["passed"],
        }

        self._save_report(report, "regression")
        self._print_gate_result(gate_result)
        return report

    async def run_gating(self, threshold: float = 0.6) -> dict[str, Any]:
        """门禁评测：关键用例 + 严格阈值"""
        logger.info("=" * 60)
        logger.info(f"CI 门禁评测开始 (threshold={threshold})")
        logger.info("=" * 60)

        from app.eval.ragas_evaluator import ragas_evaluator

        # 只测 easy + medium（核心能力）
        component_results = await ragas_evaluator.evaluate_component(
            categories=["easy", "medium"]
        )
        summary = component_results.get("summary", {})

        thresholds = {
            k: max(v, threshold) if isinstance(v, float) else v
            for k, v in CI_THRESHOLDS["gating"].items()
        }

        gate_result = self._check_gate(summary, thresholds)

        report = {
            "mode": "gating",
            "threshold": threshold,
            "timestamp": time.time(),
            "total_cases": component_results.get("total_cases", 0),
            "summary": summary,
            "gate_check": gate_result,
            "passed": gate_result["passed"],
        }

        self._save_report(report, "gating")
        self._print_gate_result(gate_result)
        return report

    async def run_smoke(self) -> dict[str, Any]:
        """冒烟评测：5 个核心用例快速验证"""
        logger.info("=" * 60)
        logger.info("CI 冒烟评测开始")
        logger.info("=" * 60)

        from app.eval.ragas_evaluator import ragas_evaluator

        # 只测 easy 类别
        component_results = await ragas_evaluator.evaluate_component(
            categories=["easy"]
        )
        summary = component_results.get("summary", {})

        gate_result = self._check_gate(summary, CI_THRESHOLDS["smoke"])

        report = {
            "mode": "smoke",
            "timestamp": time.time(),
            "total_cases": component_results.get("total_cases", 0),
            "summary": summary,
            "gate_check": gate_result,
            "passed": gate_result["passed"],
        }

        self._save_report(report, "smoke")
        self._print_gate_result(gate_result)
        return report

    # ──────────────── 全模式（E2E + 组件）────────────────

    async def run_full(
        self,
        max_e2e_cases: int = 10,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """完整评测：组件 + E2E RAGAS"""
        from app.eval.ragas_evaluator import ragas_evaluator

        logger.info("运行组件评测...")
        component_result = await ragas_evaluator.evaluate_component()

        logger.info(f"运行 E2E RAGAS 评测（最多 {max_e2e_cases} 个用例）...")
        e2e_result = await ragas_evaluator.evaluate_e2e(
            categories=["easy", "medium"],
            metrics=metrics,
            max_cases=max_e2e_cases,
        )

        report = {
            "mode": "full",
            "timestamp": time.time(),
            "component": component_result.get("summary", {}),
            "ragas_e2e": e2e_result.get("summary", {}),
            "total_cases_component": component_result.get("total_cases", 0),
            "total_cases_e2e": e2e_result.get("total_cases", 0),
        }

        self._save_report(report, "full")
        return report

    # ──────────────── 门禁检查 ────────────────

    def _check_gate(
        self, summary: dict, thresholds: dict
    ) -> dict[str, Any]:
        """检查评测结果是否通过门禁"""
        checks = []
        all_passed = True

        for metric, threshold in thresholds.items():
            actual = summary.get(metric)
            if actual is None:
                continue
            passed = actual >= threshold
            if not passed:
                all_passed = False
            checks.append({
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "passed": passed,
            })

        return {
            "passed": all_passed,
            "checks": checks,
            "summary": (
                "全部通过" if all_passed
                else f"{sum(1 for c in checks if not c['passed'])}/{len(checks)} 未通过"
            ),
        }

    def _print_gate_result(self, gate: dict):
        """打印门禁结果"""
        status = "PASSED" if gate["passed"] else "FAILED"
        logger.info(f"\n{'='*40}")
        logger.info(f"CI Gate: {status}")
        logger.info(f"{'='*40}")
        for check in gate.get("checks", []):
            icon = "[PASS]" if check["passed"] else "[FAIL]"
            logger.info(
                f"  {icon} {check['metric']}: "
                f"{check['actual']:.2%} >= {check['threshold']:.2%}"
            )
        logger.info(f"\n{gate['summary']}")

    def _save_report(self, report: dict, prefix: str):
        """保存评测报告"""
        ts = int(time.time())
        # JSON
        json_path = self.output_dir / f"ci_{prefix}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {json_path}")

        # Markdown
        md_path = self.output_dir / f"ci_{prefix}_{ts}.md"
        self._generate_markdown_report(report, md_path)
        logger.info(f"Markdown 报告: {md_path}")

    def _generate_markdown_report(self, report: dict, path: Path):
        """生成 Markdown 格式报告"""
        lines = [
            f"# CI 评测报告 — {report.get('mode', 'unknown')}",
            "",
            f"- **时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.get('timestamp', 0)))}",
            f"- **模式**: {report.get('mode', 'N/A')}",
            f"- **用例数**: {report.get('total_cases', 'N/A')}",
            f"- **门禁结果**: {'PASSED' if report.get('passed', False) else 'FAILED'}",
            "",
            "## 评测指标",
            "",
        ]

        summary = report.get("summary", {})
        if summary:
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    lines.append(f"| {k} | {v:.2%} |")
                elif isinstance(v, dict):
                    continue  # 嵌套字典单独处理
                else:
                    lines.append(f"| {k} | {v} |")

        gate = report.get("gate_check", {})
        if gate.get("checks"):
            lines.append("")
            lines.append("## 门禁检查")
            lines.append("")
            lines.append("| 指标 | 实际值 | 阈值 | 结果 |")
            lines.append("|------|--------|------|------|")
            for check in gate["checks"]:
                icon = "PASS" if check["passed"] else "FAIL"
                lines.append(
                    f"| {check['metric']} | {check['actual']:.2%} | "
                    f"{check['threshold']:.2%} | {icon} |"
                )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ──────────────── CLI ────────────────

async def main():
    parser = argparse.ArgumentParser(description="OnCall CI 评测管道")
    parser.add_argument(
        "--mode", choices=["regression", "gating", "smoke", "full"],
        default="smoke", help="评测模式"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="门禁阈值 (仅 gating 模式)"
    )
    parser.add_argument(
        "--max-e2e", type=int, default=10,
        help="E2E 最大用例数 (仅 full 模式)"
    )
    parser.add_argument(
        "--metrics", nargs="*",
        default=None,
        help="RAGAS 指标 (仅 full 模式)"
    )
    parser.add_argument(
        "--output-dir", default="eval/results",
        help="输出目录"
    )

    args = parser.parse_args()
    runner = CIEvalRunner(output_dir=args.output_dir)

    if args.mode == "regression":
        await runner.run_regression()
    elif args.mode == "gating":
        await runner.run_gating(threshold=args.threshold)
    elif args.mode == "smoke":
        await runner.run_smoke()
    elif args.mode == "full":
        await runner.run_full(max_e2e_cases=args.max_e2e, metrics=args.metrics)

    logger.info("CI 评测完成")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
