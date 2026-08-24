"""Planner × 长期记忆 集成测试：语义记忆注入 experience_context（经验复用闭环）"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from app.services.memory.types import MemoryItem, MemoryType

# 注意：app.agent.aiops.__init__ 把函数 planner 重导出到包属性上，
# 会遮蔽 `import ... as planner_mod` 的模块绑定，必须经 importlib 取真实模块
PLANNER_MODULE = importlib.import_module("app.agent.aiops.planner")

# 用真实 Plan 类构造返回值（planner 内部 isinstance 依据的是本模块的类）
Plan = PLANNER_MODULE.Plan


def make_capturing_llm(plan: Plan, captured: list) -> MagicMock:
    """结构化输出替身：返回固定 Plan 并记录链路入参（对齐 test_aiops_workflow 模式）"""

    def _next(*args, **kwargs):
        captured.append((args, kwargs))
        return plan

    llm = MagicMock(name="FakeChatModel")
    llm.with_structured_output = MagicMock(return_value=RunnableLambda(_next))
    return llm


def _captured_text(captured: list) -> str:
    """把捕获到的入参全部字符串化（容忍 PromptValue/dict 等不同形态）"""
    parts = []
    for args, kwargs in captured:
        parts.extend(repr(a) for a in args)
        parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    return "".join(parts)


@pytest.fixture
def fake_memory(monkeypatch):
    import app.services.memory as memory_pkg

    memory = MagicMock()
    memory.enabled = True
    memory.recall = AsyncMock(
        return_value=[
            MemoryItem(
                type=MemoryType.SEMANTIC,
                content="历史 OOM 事故根因是内存泄漏",
                importance=0.8,
            ),
        ]
    )
    monkeypatch.setattr(memory_pkg, "memory_service", memory)
    return memory


@pytest.fixture
def planner_env(monkeypatch):
    """屏蔽 planner 的全部外部协作者，只留纯逻辑"""
    router = MagicMock()
    router.route = AsyncMock(return_value=("DIAGNOSTIC", ["内存"]))
    router.get_retrieval_strategy.return_value = {
        "use_rag": False,
        "use_hyde": False,
        "use_kg": False,
        "rag_top_k": 3,
    }
    monkeypatch.setattr(PLANNER_MODULE, "query_router", router)

    tools_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(PLANNER_MODULE, "get_mcp_tools", tools_mock)
    return PLANNER_MODULE


class TestPlannerMemoryInjection:
    @pytest.mark.asyncio
    async def test_semantic_memory_injected_into_experience_context(self, planner_env, fake_memory):
        captured: list = []
        plan = Plan(steps=["查询日志", "生成报告"])

        with patch.object(planner_env, "LLMFactory") as factory_mock:
            factory_mock.create_chat_model.return_value = make_capturing_llm(plan, captured)
            result = await planner_env.planner({"input": "分析内存告警"})

        assert result["plan"] == ["查询日志", "生成报告"]
        # 记忆召回发生且带正确类型过滤（语义 + 程序记忆）
        fake_memory.recall.assert_awaited_once()
        assert fake_memory.recall.await_args.kwargs["types"] == [
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
        ]
        # 注入进 prompt 变量
        text = _captured_text(captured)
        assert "历史记忆" in text
        assert "内存泄漏" in text

    @pytest.mark.asyncio
    async def test_no_memory_block_when_disabled(self, planner_env, fake_memory):
        fake_memory.enabled = False
        captured: list = []

        with patch.object(planner_env, "LLMFactory") as factory_mock:
            factory_mock.create_chat_model.return_value = make_capturing_llm(
                Plan(steps=["s1"]), captured
            )
            await planner_env.planner({"input": "任务"})

        fake_memory.recall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_failure_degrades_gracefully(self, planner_env, fake_memory):
        fake_memory.recall.side_effect = RuntimeError("嵌入熔断")
        captured: list = []

        with patch.object(planner_env, "LLMFactory") as factory_mock:
            factory_mock.create_chat_model.return_value = make_capturing_llm(
                Plan(steps=["s1"]), captured
            )
            result = await planner_env.planner({"input": "任务"})

        # 主流程不受影响，正常出计划
        assert result["plan"] == ["s1"]
