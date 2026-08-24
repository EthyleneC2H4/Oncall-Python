"""Prompt 版本回归测试

在新 Prompt 版本部署前，通过 A/B 对比旧版 vs 新版 Prompt 的评测分数，
自动检测退化并生成对比报告。

使用方式:
    # 对比 baseline（当前版本）vs candidate（新版本）
    python -m app.eval.prompt_regression \
        --baseline prompts/ \
        --candidate prompts_v2/ \
        --output eval/results/prompt_diff_$(date +%s).json

    # 只对比特定 Prompt
    python -m app.eval.prompt_regression \
        --baseline prompts/system_prompt_v1.yaml \
        --candidate prompts_v2/system_prompt_v1.yaml \
        --prompt-name system_prompt

原理:
    1. 加载 baseline 和 candidate 的 Prompt 文件
    2. 对每个评测用例，分别用两个版本生成回答
    3. 对两组回答分别运行 LLM Judge (Faithfulness + Relevancy)
    4. 对比分数差异，标记退化 (score_diff < -0.1)
    5. 生成 diff 报告


    pip install pyyaml
    python -m app.eval.prompt_regression --baseline prompts/ --candidate prompts/ --output /tmp/test_diff.json --max-cases 2
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory


class PromptRegressionRunner:
    """Prompt 版本回归评测器"""

    def __init__(
        self,
        baseline_dir: str,
        candidate_dir: str,
        output_path: str,
        prompt_name: str | None = None,
        max_cases: int = 20,
    ):
        self.baseline_dir = Path(baseline_dir)
        self.candidate_dir = Path(candidate_dir)
        self.output_path = Path(output_path)
        self.prompt_name = prompt_name
        self.max_cases = max_cases

    async def run(self) -> dict[str, Any]:
        """执行回归对比评测"""
        logger.info("=" * 60)
        logger.info("Prompt 版本回归评测")
        logger.info(f"  Baseline: {self.baseline_dir}")
        logger.info(f"  Candidate: {self.candidate_dir}")
        logger.info("=" * 60)

        # 1. 加载评测用例
        cases = self._load_cases()
        logger.info(f"加载 {len(cases)} 个评测用例（最多 {self.max_cases} 个）")

        # 2. 逐用例对比
        results = []
        for tc in cases[: self.max_cases]:
            diff = await self._evaluate_case_diff(tc)
            results.append(diff)
            logger.info(
                f"  {tc['id']}: faithfulness={diff.get('faithfulness_diff', 'N/A')}, "
                f"relevancy={diff.get('relevancy_diff', 'N/A')}"
            )

        # 3. 汇总
        summary = self._compute_summary(results)

        report = {
            "baseline": str(self.baseline_dir),
            "candidate": str(self.candidate_dir),
            "timestamp": time.time(),
            "total_cases": len(results),
            "case_results": results,
            "summary": summary,
        }

        # 4. 保存
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {self.output_path}")

        # 5. 打印结论
        self._print_summary(summary)
        return report

    def _load_cases(self) -> list[dict]:
        """加载评测用例（经 dataset_registry 兼容版本信封）"""
        from app.eval.dataset_registry import read_cases

        dataset_dir = Path("eval/datasets")
        cases = []
        for filename in ["diagnostic_cases.json", "negative_cases.json"]:
            filepath = dataset_dir / filename
            if filepath.exists():
                cases.extend(read_cases(filepath))
        return cases

    async def _evaluate_case_diff(self, test_case: dict) -> dict:
        """单个用例的 A/B 对比"""
        query = test_case["query"]
        reference = test_case.get("reference", "")

        # 使用 baseline prompt 生成
        baseline_answer = await self._generate_with_prompt(query, str(self.baseline_dir))
        # 使用 candidate prompt 生成
        candidate_answer = await self._generate_with_prompt(query, str(self.candidate_dir))

        # LLM Judge 评测
        from app.eval.llm_judge import llm_judge

        baseline_judge = await llm_judge.judge_full(query, reference, baseline_answer)
        candidate_judge = await llm_judge.judge_full(query, reference, candidate_answer)

        b_faith = baseline_judge.get("faithfulness", {}).get("score", 0)
        c_faith = candidate_judge.get("faithfulness", {}).get("score", 0)
        b_rel = baseline_judge.get("relevancy", {}).get("score", 0)
        c_rel = candidate_judge.get("relevancy", {}).get("score", 0)

        return {
            "id": test_case["id"],
            "category": test_case.get("category", ""),
            "query": query,
            "baseline": {
                "answer": baseline_answer[:500],
                "faithfulness": b_faith,
                "relevancy": b_rel,
            },
            "candidate": {
                "answer": candidate_answer[:500],
                "faithfulness": c_faith,
                "relevancy": c_rel,
            },
            "faithfulness_diff": round(c_faith - b_faith, 2),
            "relevancy_diff": round(c_rel - b_rel, 2),
            "degraded": (c_faith < b_faith - 1) or (c_rel < b_rel - 1),
        }

    async def _generate_with_prompt(self, query: str, prompt_dir: str) -> str:
        """使用指定 Prompt 目录生成回答"""
        # P5 修复：经 PromptManager 组合渲染（persona/rules 块 + 正文），
        # 与生产链路一致——直接读 YAML content 会漏掉块组合，评测对象残缺
        system_prompt = ""
        prompt_path = Path(prompt_dir) / "system_prompt_v1.yaml"
        if prompt_path.exists():
            try:
                from app.core.prompt_manager import PromptManager

                system_prompt = PromptManager(prompts_dir=prompt_dir).render_composed(
                    "system_prompt"
                )
            except Exception as e:
                logger.warning(f"组合渲染失败，回退裸 content: {e}")
                try:
                    import yaml

                    with open(prompt_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    system_prompt = data.get("content", "")
                except Exception:
                    pass

        prompt = (
            (f"{system_prompt}\n\n" f"## 用户问题\n{query}\n\n" "## 回答\n")
            if system_prompt
            else query
        )

        try:
            llm = LLMFactory.create_chat_model(
                streaming=False,
                model=config.rag_model,
                temperature=0,
            )
            result = await llm.ainvoke(prompt)
            return str(result.content)
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"生成失败: {e}"

    def _compute_summary(self, results: list[dict]) -> dict:
        """计算汇总"""
        total = len(results)
        if total == 0:
            return {}

        degraded_count = sum(1 for r in results if r.get("degraded", False))
        improved_count = sum(
            1
            for r in results
            if r.get("faithfulness_diff", 0) > 1 or r.get("relevancy_diff", 0) > 1
        )

        avg_faith_diff = sum(r.get("faithfulness_diff", 0) for r in results) / total
        avg_rel_diff = sum(r.get("relevancy_diff", 0) for r in results) / total

        return {
            "total_cases": total,
            "degraded_count": degraded_count,
            "improved_count": improved_count,
            "stable_count": total - degraded_count - improved_count,
            "avg_faithfulness_diff": round(avg_faith_diff, 2),
            "avg_relevancy_diff": round(avg_rel_diff, 2),
            "verdict": (
                "PASS"
                if degraded_count == 0 and avg_faith_diff >= 0
                else "WARN" if degraded_count <= 2 else "FAIL"
            ),
        }

    def _print_summary(self, summary: dict):
        """打印汇总结论"""
        verdict = summary.get("verdict", "UNKNOWN")
        icon = {
            "PASS": "[PASS]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]",
            "UNKNOWN": "[????]",
        }.get(verdict, "[????]")

        print(f"\n{'='*50}")
        print(f"Prompt 回归评测结论: {icon} {verdict}")
        print(f"{'='*50}")
        print(f"  总用例数: {summary.get('total_cases', '?')}")
        print(f"  退化: {summary.get('degraded_count', '?')} 个")
        print(f"  提升: {summary.get('improved_count', '?')} 个")
        print(f"  持平: {summary.get('stable_count', '?')} 个")
        print(f"  Faithfulness 平均变化: {summary.get('avg_faithfulness_diff', '?'):+.2f}")
        print(f"  Relevancy 平均变化: {summary.get('avg_relevancy_diff', '?'):+.2f}")


# ──────────────── CLI ────────────────


async def main():
    parser = argparse.ArgumentParser(description="OnCall Prompt 版本回归评测")
    parser.add_argument("--baseline", required=True, help="Baseline Prompt 目录 (当前版本)")
    parser.add_argument("--candidate", required=True, help="Candidate Prompt 目录 (新版本)")
    parser.add_argument("--output", default="eval/results/prompt_diff.json", help="输出报告路径")
    parser.add_argument("--prompt-name", default=None, help="仅对比指定 Prompt (如 system_prompt)")
    parser.add_argument("--max-cases", type=int, default=20, help="最大对比用例数 (默认 20)")

    args = parser.parse_args()
    runner = PromptRegressionRunner(
        baseline_dir=args.baseline,
        candidate_dir=args.candidate,
        output_path=args.output,
        prompt_name=args.prompt_name,
        max_cases=args.max_cases,
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
