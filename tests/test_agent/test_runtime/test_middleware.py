"""TokenTrimMiddleware 单测：预算裁剪 / 孤儿 ToolMessage 修复 / 悬空 tool_calls 剥离
/ tool_calls 参数 token 计入（回归 #17）"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.runtime.middleware import (
    TokenTrimMiddleware,
    _single_message_tokens,
    repair_tool_call_continuity,
)


def _big(text: str) -> str:
    """生成足够大的中文文本（约 1.5 tok/字符）"""
    return text * 50


class TestRepairToolCallContinuity:
    def test_orphan_tool_message_dropped(self):
        msgs = [
            SystemMessage("sys"),
            HumanMessage("问题"),
            ToolMessage(content="孤儿结果", tool_call_id="call-1"),  # 父 AI 已被裁掉
            HumanMessage("追问"),
        ]
        repaired = repair_tool_call_continuity(msgs)
        assert all(not isinstance(m, ToolMessage) for m in repaired)
        assert len(repaired) == 3

    def test_paired_tool_call_kept(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "query_logs", "args": {}, "id": "call-1"}],
        )
        tool = ToolMessage(content="日志结果", tool_call_id="call-1")
        msgs = [SystemMessage("sys"), HumanMessage("查日志"), ai, tool]
        assert repair_tool_call_continuity(msgs) == msgs

    def test_dangling_tool_calls_stripped_content_kept(self):
        """AIMessage 的调用没有对应 ToolMessage（响应被裁掉）→ 剥离 tool_calls"""
        dangling = AIMessage(
            content="推理过程",
            tool_calls=[{"name": "get_metrics", "args": {}, "id": "call-x"}],
        )
        msgs = [
            SystemMessage("sys"),
            dangling,
            ToolMessage(content="", tool_call_id="call-y"),  # id 不匹配 → 孤儿
        ]
        repaired = repair_tool_call_continuity(msgs)
        repaired_ai = [m for m in repaired if isinstance(m, AIMessage)][0]
        assert repaired_ai.content == "推理过程"
        assert repaired_ai.tool_calls == []
        # 原对象未被就地修改
        assert len(dangling.tool_calls) == 1


class TestTokenTrimMiddleware:
    def _trim(self, messages, max_tokens=200):
        middleware = TokenTrimMiddleware(max_tokens=max_tokens)
        return middleware._trim(messages)

    def test_under_budget_noop(self):
        msgs = [SystemMessage("sys"), HumanMessage("短问题")]
        assert self._trim(msgs, max_tokens=10_000) is msgs  # 原样返回（含同一性）

    def test_over_budget_keeps_system_and_recent(self):
        msgs = [
            SystemMessage("系统提示词必须保留"),
            *[HumanMessage(_big(f"历史消息{i}")) for i in range(10)],
            HumanMessage("最新问题"),
        ]
        trimmed = self._trim(msgs, max_tokens=300)
        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[-1].content == "最新问题"
        assert len(trimmed) < len(msgs)

    def test_trim_repair_orphans_from_real_scenario(self):
        """裁掉父 AIMessage 后产生的孤儿 ToolMessage 被修复"""
        msgs = [
            SystemMessage("sys"),
            *[HumanMessage(_big(f"旧历史{i}")) for i in range(6)],
            AIMessage("", tool_calls=[{"name": "f", "args": {}, "id": "old-call"}]),
            ToolMessage(content="旧结果", tool_call_id="old-call"),
            HumanMessage(_big("又一段长历史")),
            AIMessage(
                "最近推理",
                tool_calls=[{"name": "g", "args": {}, "id": "new-call"}],
            ),
            ToolMessage(content="新结果", tool_call_id="new-call"),
            HumanMessage("当前问题"),
        ]
        trimmed = self._trim(msgs, max_tokens=400)
        # 不允许存在任何孤儿工具结果
        requested = set()
        for m in trimmed:
            if isinstance(m, AIMessage):
                requested |= {str(tc.get("id")) for tc in (m.tool_calls or [])}
        for m in trimmed:
            if isinstance(m, ToolMessage):
                assert str(m.tool_call_id) in requested

    async def test_awrap_model_call_trims_and_forwards(self):
        captured: dict = {}

        async def handler(request):
            captured["messages"] = request.messages
            return "模型输出"

        class FakeRequest:
            def __init__(self, messages):
                self.messages = messages

            def override(self, **kwargs):
                captured["override"] = kwargs
                return FakeRequest(kwargs["messages"])

        middleware = TokenTrimMiddleware(max_tokens=100)
        big_request = FakeRequest(
            [SystemMessage("sys"), *[HumanMessage(_big(f"h{i}")) for i in range(8)]]
        )
        result = await middleware.awrap_model_call(big_request, handler)
        assert result == "模型输出"
        assert "messages" in captured["override"]
        assert len(captured["override"]["messages"]) < 9


class TestToolCallTokenCounting:
    """回归 #17：AIMessage.tool_calls（含 JSON 参数）不计入 content，
    大参数工具调用若不单独计数会显著低估历史占用"""

    def test_tool_calls_add_tokens_over_content_only(self):
        big_args = {"query": "x" * 2000}
        ai = AIMessage(content="", tool_calls=[{"name": "search", "args": big_args, "id": "c1"}])
        bare = AIMessage(content="")
        tokens_with = _single_message_tokens(ai)
        tokens_bare = _single_message_tokens(bare)
        # 参数体（2000+ 字符）必须被计入，而非只算空 content + 角色开销
        assert tokens_with > tokens_bare + 500

    def test_counted_tokens_match_json_serialization(self):
        args = {"service": "payment", "filter": "y" * 400}
        ai = AIMessage(
            content="推理", tool_calls=[{"name": "query_logs", "args": args, "id": "c2"}]
        )
        from app.core.token_budget import token_budget_manager

        # 以消息规范化后的 tool_calls 为准（langchain 会补充 type 字段）
        expected_extra = token_budget_manager.estimate_tokens(
            json.dumps(ai.tool_calls, ensure_ascii=False)
        )
        bare = AIMessage(content="推理")
        assert (
            _single_message_tokens(ai) - _single_message_tokens(bare) == expected_extra
        )

    def test_large_args_ai_message_survives_trim_priority(self):
        """带大参数的 AI 消息在预算计算中真实占位：小预算下裁剪量与计数一致"""
        huge_call = AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {"q": "z" * 3000}, "id": "big"}],
        )
        msgs = [HumanMessage("问"), huge_call]
        total = sum(_single_message_tokens(m) for m in msgs)
        trimmed = TokenTrimMiddleware(max_tokens=total // 2)._trim(msgs)
        trimmed_total = sum(_single_message_tokens(m) for m in trimmed)
        assert trimmed_total <= total // 2
