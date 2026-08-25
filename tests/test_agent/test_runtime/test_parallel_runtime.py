"""ParallelRuntime 测试

覆盖：增量事件流（STEP_START 先于 STEP_END 到达）、汇总报告、
失败 Agent 隔离、run_parallel_diagnosis 聚合。
"""

import asyncio

import pytest

from app.agent.runtime.events import EventType
from app.agent.runtime.parallel_runtime import (
    AgentFinding,
    ParallelRuntime,
    run_parallel_diagnosis,
)


class FakeAgent:
    """可编程延迟的假专业 Agent"""

    def __init__(self, name: str, result: str = "分析完成", delay: float = 0.0,
                 confidence: float = 0.7, fail: bool = False):
        self.name = name
        self._result = result
        self._delay = delay
        self.confidence = confidence
        self._fail = fail

    async def analyze(self, alert_input: str) -> str:
        if self._fail:
            raise RuntimeError(f"{self.name} 依赖不可用")
        await asyncio.sleep(self._delay)
        return f"[{self.name}] {self._result}"


class FakeSynthesizer:
    def __init__(self):
        self.calls: list = []

    async def synthesize(self, alert_input: str, findings: list) -> str:
        self.calls.append((alert_input, list(findings)))
        return f"# 综合报告（{len(findings)} 个发现）"


class TestParallelStreaming:
    @pytest.mark.asyncio
    async def test_starts_arrive_before_ends(self):
        """每个 Agent 的 STEP_START 应先于任何 STEP_END 到达（增量流语义）"""
        agents = [
            FakeAgent("log_analyst", delay=0.05),
            FakeAgent("metric_inspector", delay=0.05),
            FakeAgent("knowledge_retriever", delay=0.05),
        ]
        synth = FakeSynthesizer()
        rt = ParallelRuntime(agents=agents, synthesizer=synth)

        events = [e async for e in rt.run("CPU 告警", session_id="p1")]
        types = [e.type for e in events]

        # 3 start → 3 end → report → complete
        assert types == [
            EventType.STEP_START,
            EventType.STEP_START,
            EventType.STEP_START,
            EventType.STEP_END,
            EventType.STEP_END,
            EventType.STEP_END,
            EventType.REPORT,
            EventType.COMPLETE,
        ]

        report_ev = events[-2]
        assert report_ev.payload["report"] == "# 综合报告（3 个发现）"

        complete_ev = events[-1]
        stats = complete_ev.payload["stats"]
        assert stats["agents_succeeded"] == 3
        assert stats["agents_failed"] == 0
        assert stats["total_duration_ms"] >= 0

        # Synthesizer 收到了全部发现
        assert len(synth.calls[0][1]) == 3

    @pytest.mark.asyncio
    async def test_failed_agent_isolated(self):
        """单个 Agent 失败不影响其他 Agent 与汇总"""
        agents = [
            FakeAgent("log_analyst", fail=True),
            FakeAgent("metric_inspector", result="指标正常"),
        ]
        rt = ParallelRuntime(agents=agents, synthesizer=FakeSynthesizer())

        events = [e async for e in rt.run("告警", session_id="p2")]
        ends = [e for e in events if e.type is EventType.STEP_END]

        by_agent = {e.payload["agent"]: e.payload["finding"] for e in ends}
        assert by_agent["log_analyst"]["error"] != ""
        assert by_agent["metric_inspector"]["error"] == ""
        assert "指标正常" in by_agent["metric_inspector"]["findings"]

        complete = events[-1]
        assert complete.payload["stats"]["agents_succeeded"] == 1
        assert complete.payload["stats"]["agents_failed"] == 1

    @pytest.mark.asyncio
    async def test_early_break_cancels_workers(self):
        """消费方提前断开时应取消仍在执行的 worker"""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class SlowAgent(FakeAgent):
            async def analyze(self, alert_input: str) -> str:
                started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return "never"

        rt = ParallelRuntime(agents=[SlowAgent("slow")], synthesizer=FakeSynthesizer())

        agen = rt.run("任务", session_id="p3")
        await agen.__anext__()  # 收到第一个 STEP_START 后断开
        await agen.aclose()

        # 等待取消落地
        for _ in range(100):
            if cancelled.is_set():
                break
            await asyncio.sleep(0.01)

        assert cancelled.is_set(), "提前断开后 worker 应被取消"


@pytest.mark.asyncio
async def test_run_parallel_diagnosis_aggregates_result():
    """兼容入口应把事件流聚合为 DiagnosisResult"""
    agents = [
        FakeAgent("log_analyst", result="日志异常"),
        FakeAgent("metric_inspector", result="CPU 90%"),
    ]
    rt = ParallelRuntime(agents=agents, synthesizer=FakeSynthesizer())

    result = await run_parallel_diagnosis("内存告警", runtime=rt)

    assert result.alert_input == "内存告警"
    assert len(result.agent_findings) == 2
    assert all(isinstance(f, AgentFinding) for f in result.agent_findings)
    assert result.agents_succeeded == 2
    assert result.agents_failed == 0
    assert "综合报告" in result.synthesized_report
    assert result.degradation_level == "none"


class ExplodingSynthesizer:
    async def synthesize(self, alert_input: str, findings: list) -> str:
        raise RuntimeError("综合模型不可用")


class TestSynthesizerFailure:
    @pytest.mark.asyncio
    async def test_synthesize_error_yields_error_terminal(self):
        """综合阶段抛异常 → 以 ERROR 终止事件收尾（基类契约），不裸穿生成器"""
        agents = [FakeAgent("log_analyst"), FakeAgent("metric_inspector")]
        rt = ParallelRuntime(agents=agents, synthesizer=ExplodingSynthesizer())

        events = [e async for e in rt.run("CPU 告警", session_id="p-err")]
        types = [e.type for e in events]

        assert EventType.STEP_END in types  # worker 结果已正常产出
        assert types[-1] is EventType.ERROR  # 消费方必须收到终止事件
        assert EventType.REPORT not in types
        assert EventType.COMPLETE not in types
        assert "多 Agent 综合失败" in events[-1].payload["message"]
