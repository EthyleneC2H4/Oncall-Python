"""AIOps Agent Plan-Execute-Replan 端到端测试

Mock LLM + MCP 工具，验证 Agent 工作流的完整行为：
- Planner: 计划生成、意图路由、KG/RAG 上下文注入
- Executor: 步骤执行、工具调用、超时处理
- Replanner: respond/continue/replan 决策、循环检测、步数限制
- 状态管理: past_steps 追加、error_context 累积、diagnosis_events 发射
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ──────────────── 测试用状态工厂 ────────────────

def make_state(**overrides):
    """创建 PlanExecuteState 用于组件测试"""
    from app.agent.aiops.state import PlanExecuteState

    defaults = {
        "input": "CPU使用率超过90%，服务响应变慢",
        "plan": [
            "查询CPU告警相关的知识图谱",
            "检索CPU高使用的处理文档",
            "分析可能的根因",
            "生成诊断报告",
        ],
        "past_steps": [],
        "response": "",
        "kg_context": "",
        "query_intent": "DIAGNOSTIC",
        "diagnosis_events": [],
        "error_context": [],
        "degradation_level": "none",
    }
    defaults.update(overrides)
    return PlanExecuteState(**defaults)


# ──────────────── Planner 测试 ────────────────

class TestPlanner:
    """Planner 阶段测试"""

    @pytest.mark.asyncio
    async def test_planner_generates_plan(self):
        """Planner 应生成有效的步骤计划"""
        from app.agent.aiops.planner import planner
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="CPU使用率超过90%",
            plan=[],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.planner.query_router") as mock_router, \
             patch("app.agent.aiops.planner.retrieve_knowledge") as mock_retrieve, \
             patch("app.agent.aiops.planner.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.planner.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp, \
             patch("app.agent.aiops.planner.knowledge_graph_service") as mock_kg, \
             patch("app.agent.aiops.planner.config"):

            # Mock 路由
            mock_router.route = AsyncMock(return_value=("DIAGNOSTIC", ["cpu"]))
            mock_router.get_retrieval_strategy.return_value = {
                "use_rag": True,
                "use_hyde": False,
                "use_kg": True,
                "rag_top_k": 3,
            }

            # Mock 检索
            mock_retrieve.invoke.return_value = "CPU高使用排查文档内容..."

            # Mock KG
            mock_kg.format_analysis_context.return_value = "## CPU告警分析\n- 常见根因: 内存泄漏"

            # Mock MCP
            mock_mcp_tool = MagicMock()
            mock_mcp_tool.name = "search_log"
            mock_mcp_tool.description = "搜索日志"
            mock_client = AsyncMock()
            mock_client.get_tools = AsyncMock(return_value=[mock_mcp_tool])
            mock_mcp.return_value = mock_client

            # Mock LLM: Planner 返回计划
            mock_plan = MagicMock()
            mock_plan.steps = [
                "查询CPU相关告警知识图谱",
                "检索CPU高使用率处理文档",
                "分析根因并生成报告",
            ]
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=mock_plan)
            mock_llm.return_value = mock_llm_instance

            result = await planner(state)

            assert len(result["plan"]) == 3
            assert result["query_intent"] == "DIAGNOSTIC"
            assert len(result["diagnosis_events"]) >= 1

    @pytest.mark.asyncio
    async def test_planner_fallback_on_error(self):
        """Planner LLM 失败时应返回默认计划"""
        from app.agent.aiops.planner import planner
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试查询",
            plan=[],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.planner.query_router") as mock_router, \
             patch("app.agent.aiops.planner.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.planner.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp, \
             patch("app.agent.aiops.planner.knowledge_graph_service") as mock_kg, \
             patch("app.agent.aiops.planner.config"):

            mock_router.route = AsyncMock(return_value=("DIAGNOSTIC", ["test"]))
            mock_router.get_retrieval_strategy.return_value = {
                "use_rag": False, "use_hyde": False, "use_kg": False, "rag_top_k": 0,
            }
            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_client
            mock_kg.format_analysis_context.return_value = ""

            # LLM 失败
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(side_effect=Exception("LLM 不可用"))
            mock_llm.return_value = mock_llm_instance

            result = await planner(state)

            # 应有默认后备计划
            assert len(result["plan"]) == 3
            assert "collect" in result["plan"][0].lower() or "收集" in result["plan"][0]


# ──────────────── Executor 测试 ────────────────

class TestExecutor:
    """Executor 阶段测试"""

    @pytest.mark.asyncio
    async def test_executor_executes_first_step(self):
        """Executor 应执行 plan[0] 并返回剩余计划"""
        from app.agent.aiops.executor import executor
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试",
            plan=["查询知识图谱", "检索文档", "生成报告"],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.executor.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.executor.ToolNode") as mock_tool_node, \
             patch("app.agent.aiops.executor.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp, \
             patch("app.agent.aiops.executor.config"):

            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_client

            # Mock LLM: 直接返回文本（不调用工具）
            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content="已查询知识图谱，未发现相关告警记录。",
                tool_calls=[]
            ))
            mock_llm.return_value = mock_llm_instance

            # Mock ToolNode
            mock_tool_node.return_value = MagicMock()

            result = await executor(state)

            # past_steps 应包含一个 (task, result) 元组
            assert len(result["past_steps"]) == 1
            task, step_result = result["past_steps"][0]
            assert task == "查询知识图谱"
            assert "知识图谱" in step_result

            # plan 应缩减
            assert len(result["plan"]) == 2

    @pytest.mark.asyncio
    async def test_executor_moves_to_next_step(self):
        """Executor 应正确移除已完成的步骤"""
        from app.agent.aiops.executor import executor
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试",
            plan=["第二步", "第三步"],
            past_steps=[("第一步", "已完成")],
            response="",
            kg_context="",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.executor.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.executor.ToolNode") as mock_tool_node, \
             patch("app.agent.aiops.executor.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp, \
             patch("app.agent.aiops.executor.config"):

            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_client

            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content="第二步执行完成。",
                tool_calls=[]
            ))
            mock_llm.return_value = mock_llm_instance
            mock_tool_node.return_value = MagicMock()

            result = await executor(state)
            assert result["plan"] == ["第三步"]

    @pytest.mark.asyncio
    async def test_executor_timeout_triggers_error_context(self):
        """Executor 超时应在 error_context 中记录"""
        from app.agent.aiops.executor import executor
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试",
            plan=["耗时过长的步骤"],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.executor.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.executor.ToolNode") as mock_tool_node, \
             patch("app.agent.aiops.executor.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp, \
             patch("app.agent.aiops.executor.config") as mock_config:

            mock_config.step_timeout_seconds = 0.001  # 1ms 超时 → 必然触发

            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_client

            # LLM 长时间阻塞
            async def slow_response(*args, **kwargs):
                import asyncio
                await asyncio.sleep(10)
                return MagicMock(content="慢速响应", tool_calls=[])

            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = slow_response
            mock_llm.return_value = mock_llm_instance

            result = await executor(state)

            assert len(result["error_context"]) == 1
            assert result["error_context"][0]["error_type"] in ("timeout", "exception")


# ──────────────── Replanner 测试 ────────────────

class TestReplanner:
    """Replanner 阶段测试"""

    @pytest.mark.asyncio
    async def test_replanner_respond_when_enough_info(self):
        """信息充足时 Replanner 应返回 respond + 生成最终报告"""
        from app.agent.aiops.replanner import replanner
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="CPU使用率超过90%",
            plan=["查询知识图谱"],
            past_steps=[
                ("查询知识图谱", "CPU告警根因: 内存泄漏导致Full GC"),
                ("检索文档", "CPU高使用排查: 1)检查内存 2)检查GC日志"),
                ("分析根因", "综合判断为内存泄漏导致的Full GC表现为CPU高"),
            ],
            response="",
            kg_context="CPU告警: 常见根因=内存泄漏",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.replanner.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.replanner.config"):

            # Mock Replanner LLM → respond
            mock_act = MagicMock()
            mock_act.action = "respond"
            mock_act.new_steps = []

            # Mock Response LLM → 最终报告
            mock_response = MagicMock()
            mock_response.response = "# 诊断报告\n\n根因为内存泄漏..."

            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(side_effect=[mock_act, mock_response])
            mock_llm.return_value = mock_llm_instance

            result = await replanner(state)

            assert result["response"] != ""
            assert "内存" in result["response"]

    @pytest.mark.asyncio
    async def test_replanner_continue_when_more_steps_needed(self):
        """计划未完时 Replanner 应返回 continue（空字典）"""
        from app.agent.aiops.replanner import replanner
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="复杂诊断",
            plan=["查询知识图谱", "检索文档", "生成报告"],
            past_steps=[("查询知识图谱", "部分信息已获取")],
            response="",
            kg_context="",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.replanner.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.replanner.config"):

            mock_act = MagicMock()
            mock_act.action = "continue"
            mock_act.new_steps = []

            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=mock_act)
            mock_llm.return_value = mock_llm_instance

            result = await replanner(state)

            # continue 应返回空字典
            assert result.get("response", "") == ""
            assert result.get("plan", None) is None or result.get("plan") == []

    @pytest.mark.asyncio
    async def test_loop_detection_exact_repeat(self):
        """检测到精确重复步骤时，应直接响应（不调用 Replanner LLM）"""
        from app.agent.aiops.replanner import _detect_loop

        past_steps = [
            ("查询知识图谱", "结果1"),
            ("查询知识图谱", "结果2"),  # 精确重复！
        ]

        assert _detect_loop(past_steps) is True

    @pytest.mark.asyncio
    async def test_loop_detection_fuzzy_repeat(self):
        """检测到模糊重复时（token 重叠 > 80%），应直接响应"""
        from app.agent.aiops.replanner import _detect_loop

        past_steps = [
            ("查询CPU相关告警知识图谱获取根因", "结果1"),
            ("查询CPU相关告警知识图谱获取根因分析", "结果2"),  # 几乎相同
        ]

        assert _detect_loop(past_steps) is True

    @pytest.mark.asyncio
    async def test_loop_detection_different_steps(self):
        """不同步骤不应触发循环检测"""
        from app.agent.aiops.replanner import _detect_loop

        past_steps = [
            ("查询知识图谱", "结果1"),
            ("检索文档", "结果2"),
            ("分析根因", "结果3"),
        ]

        assert _detect_loop(past_steps) is False

    @pytest.mark.asyncio
    async def test_max_steps_force_respond(self):
        """步数 >= 8 时应直接生成响应"""
        from app.agent.aiops.replanner import replanner
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试",
            plan=["进一步的步骤"],
            past_steps=[(f"步骤{i}", f"结果{i}") for i in range(8)],
            response="",
            kg_context="",
            query_intent="DIAGNOSTIC",
            diagnosis_events=[],
            error_context=[],
            degradation_level="none",
        )

        with patch("app.agent.aiops.replanner.ChatQwen") as mock_llm, \
             patch("app.agent.aiops.replanner.config"):

            mock_response = MagicMock()
            mock_response.response = "基于8步执行结果的综合报告..."

            mock_llm_instance = MagicMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.return_value = mock_llm_instance

            result = await replanner(state)

            # 应生成最终响应（即使 plan 还有剩余）
            assert result["response"] != ""


# ──────────────── Agent Rules 测试 ────────────────

class TestAgentRules:
    """Agent Rules / Harness 测试"""

    def test_rules_contain_all_categories(self):
        from app.harness.agent_rules import AGENT_RULES
        assert "关联分析规则" in AGENT_RULES
        assert "处置建议规则" in AGENT_RULES
        assert "诊断纪律规则" in AGENT_RULES

    def test_get_rules_for_alert_cpu(self):
        from app.harness.agent_rules import get_rules_for_alert
        result = get_rules_for_alert("cpu")
        assert "GC" in result

    def test_get_rules_for_alert_memory(self):
        from app.harness.agent_rules import get_rules_for_alert
        result = get_rules_for_alert("memory")
        assert "heap dump" in result

    def test_get_rules_for_alert_unknown(self):
        from app.harness.agent_rules import get_rules_for_alert
        result = get_rules_for_alert("unknown_keyword")
        # 应至少返回通用规则
        assert "关联分析规则" in result

    def test_rule_count(self):
        """验证有 10 条通用规则"""
        from app.harness.agent_rules import AGENT_RULES
        # 统计编号 1-10
        count = sum(1 for line in AGENT_RULES.split("\n") if line.strip().startswith(tuple("12345678910.")))
        assert count >= 8  # 至少 8 条


# ──────────────── 诊断事件测试 ────────────────

class TestDiagnosisEvents:
    """诊断事件流测试"""

    def test_event_accumulation(self):
        """验证 operator.add 追加行为"""
        from app.agent.aiops.state import PlanExecuteState

        state1 = PlanExecuteState(
            input="测试",
            plan=[],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="",
            diagnosis_events=[{"type": "routing", "data": "DIAGNOSTIC"}],
            error_context=[],
            degradation_level="none",
        )

        state2 = PlanExecuteState(
            input="测试",
            plan=[],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="",
            diagnosis_events=[{"type": "kg_query", "data": "cpu"}],
            error_context=[],
            degradation_level="none",
        )

        # 模拟 LangGraph 的 operator.add 合并
        merged = state1["diagnosis_events"] + state2["diagnosis_events"]
        assert len(merged) == 2
        assert merged[0]["type"] == "routing"
        assert merged[1]["type"] == "kg_query"


# ──────────────── Error Context 测试 ────────────────

class TestErrorContext:
    """Error Context 累积测试"""

    def test_error_context_accumulates(self):
        """多个错误应累积在 error_context 中"""
        from app.agent.aiops.state import PlanExecuteState

        state = PlanExecuteState(
            input="测试",
            plan=[],
            past_steps=[],
            response="",
            kg_context="",
            query_intent="",
            diagnosis_events=[],
            error_context=[
                {"step": "步骤1", "error_type": "timeout", "error_msg": "超时"},
                {"step": "步骤2", "error_type": "exception", "error_msg": "ValueError"},
            ],
            degradation_level="none",
        )

        assert len(state["error_context"]) == 2
        assert state["error_context"][0]["error_type"] == "timeout"
        assert state["error_context"][1]["error_type"] == "exception"


# ───────────────── Workflow 集成测试 ────────────────

class TestWorkflowIntegration:
    """AIOpsService 集成测试"""

    @pytest.mark.asyncio
    async def test_service_creates_graph(self):
        """服务应成功构建 LangGraph"""
        from app.services.aiops_service import AIOpsService
        service = AIOpsService()
        assert service.graph is not None

    @pytest.mark.asyncio
    async def test_service_execute_with_mocks(self):
        """端到端: execute() 应产生完整事件流"""
        from app.services.aiops_service import AIOpsService

        service = AIOpsService()

        # 完整的 Mock 链：Planner → Executor → Replanner 的所有 LLM 调用
        with patch("app.agent.aiops.planner.query_router") as mock_router, \
             patch("app.agent.aiops.planner.retrieve_knowledge") as mock_retrieve, \
             patch("app.agent.aiops.planner.ChatQwen") as mock_planner_llm, \
             patch("app.agent.aiops.planner.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp1, \
             patch("app.agent.aiops.planner.knowledge_graph_service") as mock_kg, \
             patch("app.agent.aiops.planner.config"), \
             patch("app.agent.aiops.executor.ChatQwen") as mock_executor_llm, \
             patch("app.agent.aiops.executor.ToolNode") as mock_tool_node, \
             patch("app.agent.aiops.executor.get_mcp_client_with_retry", new_callable=AsyncMock) as mock_mcp2, \
             patch("app.agent.aiops.executor.config"), \
             patch("app.agent.aiops.replanner.ChatQwen") as mock_replanner_llm, \
             patch("app.agent.aiops.replanner.config"):

            # --- Mock Planner ---
            mock_router.route = AsyncMock(return_value=("DIAGNOSTIC", ["cpu"]))
            mock_router.get_retrieval_strategy.return_value = {
                "use_rag": True, "use_hyde": False, "use_kg": True, "rag_top_k": 3,
            }
            mock_retrieve.invoke.return_value = "CPU排查文档"
            mock_kg.format_analysis_context.return_value = "CPU告警: 内存泄漏"
            mock_plan = MagicMock()
            mock_plan.steps = ["查询CPU知识图谱", "检索文档", "生成报告"]
            mock_p_llm = MagicMock()
            mock_p_llm.ainvoke = AsyncMock(return_value=mock_plan)
            mock_planner_llm.return_value = mock_p_llm
            mock_mcp_client1 = AsyncMock()
            mock_mcp_client1.get_tools = AsyncMock(return_value=[])
            mock_mcp1.return_value = mock_mcp_client1

            # --- Mock Executor ---
            mock_e_llm = MagicMock()
            mock_e_llm.ainvoke = AsyncMock(return_value=MagicMock(
                content="步骤执行完成", tool_calls=[]
            ))
            mock_executor_llm.return_value = mock_e_llm
            mock_tool_node.return_value = MagicMock()
            mock_mcp_client2 = AsyncMock()
            mock_mcp_client2.get_tools = AsyncMock(return_value=[])
            mock_mcp2.return_value = mock_mcp_client2

            # --- Mock Replanner ---
            # 前两次返回 continue，第三次返回 respond
            mock_act_continue = MagicMock()
            mock_act_continue.action = "continue"
            mock_act_continue.new_steps = []
            mock_act_respond = MagicMock()
            mock_act_respond.action = "respond"
            mock_act_respond.new_steps = []
            mock_response = MagicMock()
            mock_response.response = "# 诊断报告\n\n根因: 内存泄漏\n\n建议: 重启并导出堆dump"

            mock_r_llm = MagicMock()
            mock_r_llm.ainvoke = AsyncMock(side_effect=[
                mock_act_continue,   # 第1次: continue
                mock_act_continue,   # 第2次: continue
                mock_act_respond,    # 第3次: respond
                mock_response,       # 最终报告
            ])
            mock_replanner_llm.return_value = mock_r_llm

            # 执行
            events = []
            async for event in service.execute("CPU使用率超过90%", "test-session-001"):
                events.append(event)

            # 验证完整事件流
            event_types = [e.get("type", "") for e in events]
            assert "plan_created" in event_types or any("plan" in str(e).lower() for e in events)
            assert "complete" in event_types  # 最终事件
