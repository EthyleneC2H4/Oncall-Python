"""ReAct 运行时 - Token 预算中间件

替换旧实现「硬编码保留最近 N 条消息」的截断策略：按 token 计数裁剪对话历史，
系统提示词始终保留（include_system）。

langchain_core.messages.trim_messages 不保护 tool_call 配对连续性（实测：
父 AIMessage 被裁掉后会留下孤儿 ToolMessage），本模块在裁剪后做两轮修复：
1. 删除孤儿 ToolMessage（其 tool_call_id 在保留消息中无对应 AIMessage）
2. 剥离未获响应的 AIMessage.tool_calls（避免模型收到「调用了工具却没有结果」的悬空调用）
"""

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, trim_messages
from loguru import logger

from app.core.token_budget import token_budget_manager


def _single_message_tokens(msg: BaseMessage) -> int:
    """单条消息 token 近似：文本内容 + 角色开销（4）

    AIMessage 的 tool_calls（含 JSON 参数）不计入 content，
    大参数工具调用会显著低估占用 → 单独序列化计入。
    """
    tokens = token_budget_manager.estimate_tokens(str(msg.content)) + 4
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        try:
            payload = json.dumps(tool_calls, ensure_ascii=False)
            tokens += token_budget_manager.estimate_tokens(payload)
        except (TypeError, ValueError):  # pragma: no cover - 不可序列化的畸形参数
            pass
    return tokens


def _message_tokens(messages: list[BaseMessage]) -> int:
    """list 级 token 计数器（trim_messages 的批量口径，含每条消息的角色开销）"""
    return sum(_single_message_tokens(m) for m in messages)


def repair_tool_call_continuity(messages: list[BaseMessage]) -> list[BaseMessage]:
    """修复裁剪造成的 tool_call 断裂（孤儿 ToolMessage / 悬空 tool_calls）"""
    # 第一遍：确定存活的 tool_call_id（发起方 AIMessage 仍在保留集中）
    requested_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                call_id = tc.get("id")
                if call_id:
                    requested_ids.add(str(call_id))

    survived_tool_ids = {
        str(getattr(t, "tool_call_id", ""))
        for t in messages
        if isinstance(t, ToolMessage) and str(getattr(t, "tool_call_id", "")) in requested_ids
    }

    # 第二遍：丢弃孤儿工具结果；剥离悬空调用（保留推理文本）
    repaired: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if str(getattr(msg, "tool_call_id", "")) not in survived_tool_ids:
                continue  # 孤儿工具结果：父调用已被裁掉
            repaired.append(msg)
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            unresolved = [tc for tc in msg.tool_calls if str(tc.get("id")) not in survived_tool_ids]
            if unresolved:
                # 构造新实例避免原地修改共享消息对象
                msg = AIMessage(
                    content=msg.content,
                    id=msg.id,
                    additional_kwargs=dict(msg.additional_kwargs),
                )
        repaired.append(msg)
    return repaired


class TokenTrimMiddleware(AgentMiddleware):
    """按 token 预算裁剪进入模型的对话历史（strategy=last，保留 SystemMessage）"""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        total = _message_tokens(messages)
        if total <= self.max_tokens:
            return messages
        trimmed = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            token_counter=_message_tokens,
            strategy="last",
            include_system=True,
        )
        result = repair_tool_call_continuity(trimmed)
        logger.debug(f"历史裁剪: {len(messages)} 条/{total} tok → {len(result)} 条")
        return result

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        trimmed = self._trim(list(request.messages))
        return await handler(request.override(messages=trimmed))

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        trimmed = self._trim(list(request.messages))
        return handler(request.override(messages=trimmed))
