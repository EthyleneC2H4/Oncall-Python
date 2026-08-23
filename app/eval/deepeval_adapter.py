"""DeepEval 适配器 — Agent 评测指标集成

基于 DeepEval 框架，为 OnCall 的 Agent 诊断链路提供标准化评测指标：

1. Agentic 指标（基于轨迹）:
   - TaskCompletion: Agent 是否完成诊断目标
   - ToolCorrectness: 是否选择了正确的工具
   - ArgumentCorrectness: 工具调用参数是否正确

2. 与传统 RAGAS 的区别:
   - RAGAS: 评估 RAG 管道的检索和生成质量
   - DeepEval: 评估 Agent 的推理轨迹和工具使用质量
   - 两者互补: RAGAS score ∈ [0, 1] + DeepEval score ∈ [0, 1] = 完整 Agent 质量画像

3. 使用方式:
   - pytest 集成: deepeval test run tests/
   - 编程调用: adapter.evaluate_tool_correctness(...)
   - CI 门禁: adapter.run_ci_gate(threshold=0.7)

参考:
- DeepEval Agent Metrics: https://deepeval.com/guides/guides-ai-agent-evaluation-metrics
- DeepEval Tool Correctness: https://deepeval.com/docs/metrics-tool-correctness
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class AgentTrace:
    """Agent 执行轨迹"""

    user_input: str
    plan: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tool_args: list[dict] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    final_output: str = ""
    steps_count: int = 0
    latency_ms: float = 0.0


@dataclass
class AgentEvalResult:
    """Agent 评测结果"""

    task_completion_score: float = 0.0
    tool_correctness_score: float = 0.0
    argument_correctness_score: float = 0.0
    step_efficiency_score: float = 0.0
    compound_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class DeepEvalAdapter:
    """DeepEval 评测适配器

    封装 DeepEval 的 agentic 指标，提供面向 OnCall Agent 的诊断评测能力。
    """

    def __init__(self):
        self._evaluator_llm = None
        self._available: bool = False
        self._check_availability()

    def _check_availability(self) -> bool:
        """检查 DeepEval 是否可用"""
        try:
            import deepeval  # noqa: F401

            self._available = True
            logger.info("DeepEval 可用，启用 Agentic 评测指标")
        except ImportError:
            logger.warning("DeepEval 未安装。Agentic 指标不可用。" "安装: pip install deepeval")
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    # ──────────────── 工具正确性评测 ────────────────

    async def evaluate_tool_correctness(
        self,
        user_input: str,
        tools_called: list[str],
        expected_tools: list[str],
        available_tools: list[str] | None = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """评测 Agent 工具选择的正确性

        Args:
            user_input: 用户原始输入
            tools_called: Agent 实际调用的工具列表
            expected_tools: 期望调用的工具列表
            available_tools: 所有可用工具（用于评估是否选了最优工具）
            threshold: 通过阈值

        Returns:
            {"score": float, "passed": bool, "reason": str}
        """
        if not self._available:
            return self._fallback_tool_correctness(tools_called, expected_tools, threshold)

        try:
            from deepeval.metrics import ToolCorrectnessMetric
            from deepeval.test_case import LLMTestCase, ToolCall

            # 注: 当前 deepeval 版本的 ToolCall 必填 input_parameters；
            # ToolCorrectnessMetric 不再支持 available_tools 参数（候选集约束由调用方保证）
            actual_tools = [ToolCall(name=t, input_parameters={}) for t in tools_called]
            expected = [ToolCall(name=t, input_parameters={}) for t in expected_tools]

            metric = ToolCorrectnessMetric(
                threshold=threshold,
                include_reason=True,
            )

            test_case = LLMTestCase(
                input=user_input,
                actual_output="",  # not used for tool correctness
                tools_called=actual_tools,
                expected_tools=expected,
            )

            metric.measure(test_case)
            score = float(metric.score or 0.0)
            return {
                "score": round(score, 4),
                "passed": bool(metric.is_successful()),
                "reason": str(getattr(metric, "reason", "")),
            }
        except Exception as e:
            logger.error(f"ToolCorrectness 评测失败: {e}")
            return self._fallback_tool_correctness(tools_called, expected_tools, threshold)

    def _fallback_tool_correctness(
        self,
        tools_called: list[str],
        expected_tools: list[str],
        threshold: float,
    ) -> dict:
        """降级方案：基于 Jaccard 相似度的工具正确性"""
        if not expected_tools:
            return {"score": 1.0, "passed": True, "reason": "无期望工具"}
        called_set = set(tools_called)
        expected_set = set(expected_tools)
        intersection = called_set & expected_set
        union = called_set | expected_set
        score = len(intersection) / len(union) if union else 0.0
        return {
            "score": round(score, 4),
            "passed": score >= threshold,
            "reason": f"Jaccard 相似度: {score:.2%} (={len(intersection)}/{len(union)})",
        }

    # ──────────────── 任务完成度评测 ────────────────

    async def evaluate_task_completion(
        self,
        user_input: str,
        actual_output: str,
        expected_output: str,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """评测 Agent 是否完成了诊断任务

        Args:
            user_input: 用户查询
            actual_output: Agent 实际输出
            expected_output: 期望输出（Ground Truth）
            threshold: 通过阈值
        """
        if not self._available:
            return self._fallback_task_completion(actual_output, expected_output, threshold)

        try:
            from deepeval.metrics import TaskCompletionMetric
            from deepeval.test_case import LLMTestCase

            metric = TaskCompletionMetric(
                threshold=threshold,
                include_reason=True,
            )

            test_case = LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            metric.measure(test_case)
            score = float(metric.score or 0.0)
            return {
                "score": round(score, 4),
                "passed": metric.is_successful(),
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error(f"TaskCompletion 评测失败: {e}")
            return self._fallback_task_completion(actual_output, expected_output, threshold)

    def _fallback_task_completion(
        self,
        actual_output: str,
        expected_output: str,
        threshold: float,
    ) -> dict:
        """降级方案：关键词匹配"""
        if not expected_output:
            return {"score": 1.0, "passed": True, "reason": "无期望输出"}
        # 简单的关键词重叠
        actual_words = set(actual_output)
        expected_words = set(expected_output)
        overlap = len(actual_words & expected_words)
        score = min(overlap / len(expected_words), 1.0) if expected_words else 0.0
        return {
            "score": round(score, 4),
            "passed": score >= threshold,
            "reason": f"关键词重叠: {score:.2%}",
        }

    # ──────────────── 完整 Agent 评测 ────────────────

    async def evaluate_agent_trace(
        self,
        trace: AgentTrace,
        expected_tools: list[str] | None = None,
        expected_output: str = "",
        threshold: float = 0.5,
    ) -> AgentEvalResult:
        """基于完整执行轨迹的 Agent 评测

        Args:
            trace: Agent 执行轨迹
            expected_tools: 期望调用的工具
            expected_output: 期望的诊断结论
            threshold: 通过阈值

        Returns:
            AgentEvalResult with all scores
        """
        # 1. 工具正确性
        tool_result = await self.evaluate_tool_correctness(
            user_input=trace.user_input,
            tools_called=trace.tools_called,
            expected_tools=expected_tools or [],
            threshold=threshold,
        )

        # 2. 任务完成度
        task_result = await self.evaluate_task_completion(
            user_input=trace.user_input,
            actual_output=trace.final_output,
            expected_output=expected_output,
            threshold=threshold,
        )

        # 3. 步骤效率 (简化版)
        efficiency = self._compute_step_efficiency(trace)

        # 4. 综合评分
        scores = [tool_result["score"], task_result["score"], efficiency]
        compound = sum(scores) / len(scores)

        return AgentEvalResult(
            task_completion_score=task_result["score"],
            tool_correctness_score=tool_result["score"],
            argument_correctness_score=0.0,  # 需要 schema 对比
            step_efficiency_score=efficiency,
            compound_score=round(compound, 4),
            details={
                "tool_correctness": tool_result,
                "task_completion": task_result,
                "step_efficiency": {"score": efficiency},
            },
        )

    def _compute_step_efficiency(self, trace: AgentTrace) -> float:
        """计算步骤效率（简化版）

        逻辑: 步骤越少且工具调用有用 → 效率越高
        假设 1-3 步为最优，>8 步为低效
        """
        n = trace.steps_count
        if n == 0:
            return 1.0
        if n <= 3:
            return 1.0
        if n <= 5:
            return 0.8
        if n <= 8:
            return 0.5
        return max(0.1, 1.0 - (n - 3) * 0.1)

    # ──────────────── CI 门禁 ────────────────

    def run_ci_gate(
        self,
        result: AgentEvalResult,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """CI 门禁：检查 Agent 评测结果是否通过阈值

        Args:
            result: 评测结果
            thresholds: 各维度阈值，默认 {"compound": 0.6, "tool_correctness": 0.5}

        Returns:
            {"passed": bool, "checks": [...], "summary": str}
        """
        if thresholds is None:
            thresholds = {"compound": 0.6, "tool_correctness": 0.5}

        checks = []
        all_passed = True

        name_map = {
            "compound": ("综合评分", result.compound_score),
            "tool_correctness": ("工具正确性", result.tool_correctness_score),
            "task_completion": ("任务完成度", result.task_completion_score),
            "step_efficiency": ("步骤效率", result.step_efficiency_score),
        }

        for key, (label, score) in name_map.items():
            if key in thresholds:
                passed = score >= thresholds[key]
                if not passed:
                    all_passed = False
                checks.append(
                    {
                        "check": label,
                        "score": score,
                        "threshold": thresholds[key],
                        "passed": passed,
                    }
                )

        return {
            "passed": all_passed,
            "checks": checks,
            "summary": (
                "所有门禁通过"
                if all_passed
                else f"{sum(1 for c in checks if not c['passed'])} 项未通过"
            ),
        }


# 全局单例
deepeval_adapter = DeepEvalAdapter()
