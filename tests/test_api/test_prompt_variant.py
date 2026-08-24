"""Prompt 变体贯通测试（P5-a）：X-Prompt-Variant 请求头 → service → runtime → 归因

契约：
- 合法变体名原样下发并在 done 事件附带生效名（只增不改）
- 非法请求头静默按基线处理，不打断对话
- 未登记变体回退基线；cost_tracker 分组按 runtime 实际使用的图记录
  （评审修复后归因单一事实源在 runtime.run，不在 service 预标记）
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.api.chat import _sanitize_prompt_variant


class _RecordingRuntime:
    """捕获 run() 收到的 prompt_variant 的桩 runtime"""

    def __init__(self):
        self.calls: list[dict] = []

    def _record(self, question, session_id, prompt_variant):
        self.calls.append(
            {"question": question, "session_id": session_id, "prompt_variant": prompt_variant}
        )

        async def _empty_stream():
            return
            yield  # pragma: no cover - 空 async 生成器

        return _empty_stream()

    def run(self, task, session_id="default", *, prompt_variant=None):
        return self._record(task, session_id, prompt_variant)


class TestHeaderSanitization:
    def test_valid_names_pass_through(self):
        assert _sanitize_prompt_variant("concise") == "concise"
        assert _sanitize_prompt_variant(" v2-long_name ") == "v2-long_name"

    def test_mixed_case_normalized_to_lowercase(self):
        """变体名约定全小写：请求头大小写归一，避免分组分裂"""
        assert _sanitize_prompt_variant("Concise") == "concise"
        assert _sanitize_prompt_variant(" V2-Long_Name ") == "v2-long_name"

    def test_invalid_values_rejected_to_none(self):
        assert _sanitize_prompt_variant(None) is None
        assert _sanitize_prompt_variant("") is None
        assert _sanitize_prompt_variant("a; rm -rf") is None
        assert _sanitize_prompt_variant("x" * 33) is None


class TestServiceThreading:
    @pytest.fixture()
    def service_with_stub(self, monkeypatch):
        from app.services import rag_agent_service as svc_module

        stub = _RecordingRuntime()
        monkeypatch.setattr(svc_module.rag_agent_service, "runtime", stub)
        return svc_module.rag_agent_service, stub

    async def test_valid_variant_resolved_and_forwarded(self, service_with_stub):
        svc, stub = service_with_stub
        await svc.query("CPU 高", "s1", prompt_variant="concise")
        assert stub.calls[-1]["prompt_variant"] == "concise"

    async def test_unregistered_variant_falls_back_to_base(self, service_with_stub):
        svc, stub = service_with_stub
        await svc.query("CPU 高", "s1", prompt_variant="ghost")
        assert stub.calls[-1]["prompt_variant"] in (None, "")

    async def test_no_header_runs_base(self, service_with_stub):
        svc, stub = service_with_stub
        await svc.query("CPU 高", "s1")
        assert stub.calls[-1]["prompt_variant"] in (None, "")


class _FakeTokenChunk:
    """类名对齐 runtime 白名单（AIMessageChunk）的最小 token 桩"""

    def __init__(self, text: str):
        self.content_blocks = [{"type": "text", "text": text}]


class _FakeGraph:
    """最小编译图桩：messages 通道产出一个 token 后正常结束"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def astream(self, input=None, config=None, stream_mode=None):
        yield "messages", (_FakeTokenChunk("答"), {"langgraph_node": "agent"})


class TestVariantAttribution:
    """归因单一事实源在 runtime.run：按实际生效的图记录并随 COMPLETE 下发"""

    def _install_fakes(self, monkeypatch):
        built_prompts: list[str] = []

        def fake_create_agent(model, tools=None, **kwargs):
            graph = _FakeGraph(str(kwargs.get("system_prompt", "")))
            built_prompts.append(graph.system_prompt)
            return graph

        async def no_mcp():
            return []

        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.create_agent", fake_create_agent
        )
        monkeypatch.setattr(
            "app.agent.runtime.react_runtime.get_mcp_tools", no_mcp
        )
        return built_prompts

    async def test_run_marks_actual_usage_per_session(self, monkeypatch):
        from app.agent.runtime.react_runtime import ReActRuntime
        from app.core.cost_tracker import CostTracker

        tracker = CostTracker()
        monkeypatch.setattr("app.core.cost_tracker.cost_tracker", tracker)
        self._install_fakes(monkeypatch)

        rt = ReActRuntime(tools=[], system_prompt="基础提示词")
        base_events = [e async for e in rt.run("q", session_id="sess-a")]
        variant_events = [
            e async for e in rt.run("q", session_id="sess-b", prompt_variant="concise")
        ]

        summary = tracker.get_summary()["prompt_variants"]
        assert summary["base"] == {"runs": 1, "sessions": 1}
        assert summary["concise"] == {"runs": 1, "sessions": 1}

        from app.agent.runtime.events import EventType

        complete_base = next(e for e in base_events if e.type is EventType.COMPLETE)
        assert "prompt_variant" not in complete_base.payload  # 基线不带字段
        complete_variant = next(
            e for e in variant_events if e.type is EventType.COMPLETE
        )
        assert complete_variant.payload.get("prompt_variant") == "concise"


class TestApiContract:
    def test_chat_forwards_header(self, test_app, monkeypatch):
        from app.services import rag_agent_service as svc_module

        captured: dict = {}

        async def fake_query(question, session_id, prompt_variant=None):
            captured.update(prompt_variant=prompt_variant)
            return "ok"

        monkeypatch.setattr(svc_module.rag_agent_service, "query", fake_query)
        resp = test_app.post(
            "/api/chat",
            json={"id": "s9", "question": "hi"},
            headers={"X-Prompt-Variant": "concise"},
        )
        assert resp.status_code == 200
        assert captured["prompt_variant"] == "concise"

    def test_chat_invalid_header_treated_as_base(self, test_app, monkeypatch):
        from app.services import rag_agent_service as svc_module

        captured: dict = {}

        async def fake_query(question, session_id, prompt_variant=None):
            captured.update(prompt_variant=prompt_variant)
            return "ok"

        monkeypatch.setattr(svc_module.rag_agent_service, "query", fake_query)
        resp = test_app.post(
            "/api/chat",
            json={"id": "s9", "question": "hi"},
            headers={"X-Prompt-Variant": "bad value!"},
        )
        assert resp.status_code == 200
        assert captured["prompt_variant"] is None

    def test_no_header_omits_field(self, test_app, monkeypatch):
        from app.services import rag_agent_service as svc_module

        async def fake_query(question, session_id, prompt_variant=None):
            return "ok"

        monkeypatch.setattr(svc_module.rag_agent_service, "query", fake_query)
        resp = test_app.post("/api/chat", json={"id": "s9", "question": "hi"})
        assert resp.status_code == 200


class TestRuntimeVariantAgent:
    async def test_variant_agent_built_once_and_cached(self, monkeypatch):
        """同变体第二次运行复用编译图；基线走主实例；
        变体图正文含 concise 独有标记（真实 prompts/system_prompt_v1.yaml）"""
        from app.agent.runtime.react_runtime import ReActRuntime

        rt = ReActRuntime(tools=[], system_prompt="基础提示词")

        built_prompts: list[str] = []

        def fake_create_agent(model, tools=None, **kwargs):
            built_prompts.append(str(kwargs.get("system_prompt", "")))
            return SimpleNamespace(astream=None)

        async def no_mcp():
            return []

        with (
            patch("app.agent.runtime.react_runtime.create_agent", side_effect=fake_create_agent),
            patch("app.agent.runtime.react_runtime.get_mcp_tools", no_mcp),
        ):
            agent_a, used_a = await rt._agent_for_variant("concise")
            agent_b, used_b = await rt._agent_for_variant("concise")
            base_agent, used_base = await rt._agent_for_variant(None)

        assert agent_a is agent_b  # 缓存命中，create_agent 只跑一次/变体
        assert used_a == used_b == "concise"  # 实际生效名回传给归因
        assert base_agent is rt.agent and used_base == ""
        assert len(built_prompts) == 2  # 基线一次 + concise 一次
        assert built_prompts[0] == "基础提示词"
        # 变体图携带组合后的 concise 正文独有标记（评审 L 加固：
        # 此前只断言「不等于基线」，无法发现变体渲染悄悄退化为基线）
        assert "不超过两句话" in built_prompts[1]

    async def test_unregistered_variant_returns_main_instance(self, monkeypatch):
        """未登记变体在 runtime 层也回退主实例且归因为空——
        不为等价基线正文重复编译图、不计入变体分组"""
        from app.agent.runtime.react_runtime import ReActRuntime

        rt = ReActRuntime(tools=[], system_prompt="基础提示词")
        builds: list[str] = []

        def fake_create_agent(model, tools=None, **kwargs):
            builds.append(str(kwargs.get("system_prompt", "")))
            return SimpleNamespace(astream=None)

        async def no_mcp():
            return []

        with (
            patch("app.agent.runtime.react_runtime.create_agent", side_effect=fake_create_agent),
            patch("app.agent.runtime.react_runtime.get_mcp_tools", no_mcp),
        ):
            agent, used = await rt._agent_for_variant("ghost-variant")

        assert agent is rt.agent and used == ""
        assert len(builds) == 1  # 只有基线这一次构建

    async def test_cache_rebuilds_when_rendered_text_changes(self, monkeypatch):
        """评审 #7：块热加载导致渲染文本变化时必须重建变体图，
        否则变体永远持有构建期的提示词快照"""
        from app.agent.runtime.react_runtime import ReActRuntime
        from app.core.prompt_manager import prompt_manager

        rt = ReActRuntime(tools=[], system_prompt="基础提示词")
        texts = iter(["第一版变体文本", "第一版变体文本", "第二版变体文本"])
        monkeypatch.setattr(
            prompt_manager,
            "render_composed",
            lambda *args, **kwargs: next(texts),
        )
        # effective_variant 走真实模板（concise 已登记），不受 render 桩影响

        builds: list[str] = []

        def fake_create_agent(model, tools=None, **kwargs):
            builds.append(str(kwargs.get("system_prompt", "")))
            return SimpleNamespace(astream=None)

        async def no_mcp():
            return []

        with (
            patch("app.agent.runtime.react_runtime.create_agent", side_effect=fake_create_agent),
            patch("app.agent.runtime.react_runtime.get_mcp_tools", no_mcp),
        ):
            first, _ = await rt._agent_for_variant("concise")  # 构建 v1
            second, _ = await rt._agent_for_variant("concise")  # 文本相同 → 缓存命中
            third, _ = await rt._agent_for_variant("concise")  # 文本已变 → 重建 v2

        assert first is second
        assert third is not first
        assert [b for b in builds if b.startswith(("第一版", "第二版"))] == [
            "第一版变体文本",
            "第二版变体文本",
        ]

    async def test_base_run_uses_main_instance_not_cache(self):
        from app.agent.runtime.react_runtime import ReActRuntime

        rt = ReActRuntime(tools=[], system_prompt="基础提示词")

        async def no_mcp():
            return []

        with patch("app.agent.runtime.react_runtime.get_mcp_tools", no_mcp):
            agent, used = await rt._agent_for_variant("")
        assert agent is rt.agent and used == ""
