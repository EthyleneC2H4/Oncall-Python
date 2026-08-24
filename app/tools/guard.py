"""工具执行咽喉 —— 所有「计划内直接工具调用」的统一入口

把 tool_registry 里沉睡的权限校验 / 参数验证 / 审计声明接入真实执行路径：

    guarded_call(tool, args)
        1. 未注册工具：按 READ_ONLY 放行但强制审计（兼容动态 MCP 工具）；
           参数键名/字符串值携带破坏性意图时拒绝（启发式纵深防御，
           真正的高风险操作必须先在 tool_registry 登记才可被确认门管控）
        2. check_permission：requires_confirmation 的高风险操作不执行，
           登记 pending action 并返回 needs_confirmation + action_id，
           由 POST /api/actions/{id}/approve|reject 人工裁决后补执行
        3. validate_params：schema 校验失败直接拒绝
        4. 执行 + 审计 + 时延记录

GuardResult 永不抛异常：任何失败都折叠为 ok=False 的结果对象。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.core.audit import audit_logger
from app.core.trace_sink import tool_trace_sink
from app.services.pending_actions import (
    ActionStatus,
    PendingAction,
    get_pending_action_store,
)
from app.tools.tool_registry import RiskLevel, tool_registry


@dataclass
class GuardResult:
    """guarded_call 的结果对象"""

    ok: bool
    value: Any = None  # 执行成功时的输出文本
    error: str = ""  # 拒绝原因或执行错误
    needs_confirmation: bool = False  # True 时调用方应向用户展示审批链接
    action_id: str = ""  # 待审动作 id（needs_confirmation=True 时非空）
    meta: dict[str, Any] = field(default_factory=dict)


async def guarded_call(
    tool: Any,
    args: dict[str, Any] | None = None,
    *,
    request_id: str = "",
    session_id: str = "",
    approved_action_id: str = "",
) -> GuardResult:
    """经权限/参数/确认门三道关卡执行一次工具调用（永不抛异常）

    Args:
        approved_action_id: 补执行凭证——必须是一条已获人工批准
            （approved/executed 认领态）的待审动作 id，凭此跳过确认门；
            凭证无效时拒绝执行而非放行。其余调用方不得传值。
    """
    name = getattr(tool, "name", str(tool))
    call_args = args or {}
    started = time.perf_counter()
    meta = tool_registry.get(name)
    unregistered = meta is None

    try:
        # ── 关卡 1：权限 ──
        if meta is not None:
            allowed, reason = tool_registry.check_permission(name)
            needs_confirm = not allowed and meta.requires_confirmation
            if needs_confirm:
                if not approved_action_id:
                    return await _propose_confirmation(
                        name, call_args, reason=reason,
                        request_id=request_id, session_id=session_id,
                    )
                # 凭证校验：只认已批准/已被补执行认领的动作
                claimed = get_pending_action_store().get(approved_action_id)
                if claimed is None or claimed.tool_name != name or claimed.status not in (
                    ActionStatus.APPROVED,
                    ActionStatus.EXECUTED,
                ):
                    _audit(name, call_args, "rejected", started, request_id,
                           error="无效的补执行凭证")
                    return GuardResult(
                        ok=False,
                        error="补执行凭证无效（动作不存在、未批准或与工具不符）",
                    )
                logger.info(f"{name} 凭动作 {approved_action_id} 的人工批准跳过确认门")
            elif not allowed:
                return _denied(name, reason, started)

        # ── 关卡 2：参数校验（未注册工具无 schema，跳过）──
        if meta is not None:
            valid, err = tool_registry.validate_params(name, call_args)
            if not valid:
                _audit(name, call_args, "rejected", started, request_id, error=err)
                return GuardResult(ok=False, error=f"参数校验失败: {err}")
        elif _looks_destructive(call_args):
            # 防御纵深：未注册却带明显破坏性意图的参数形状，拒绝并审计
            reason = "未注册工具携带疑似破坏性参数，已拒绝（请先在 tool_registry 登记）"
            _audit(name, call_args, "rejected", started, request_id, error=reason)
            return GuardResult(ok=False, error=reason)

        # ── 关卡 3：执行 ──
        result = await _invoke_tool(tool, call_args)
        status = "success"
        latency_ms = (time.perf_counter() - started) * 1000
        _audit(name, call_args, status, started, request_id)
        # 完整实参另落评测痕迹（与审计分离：审计只存摘要）
        tool_trace_sink.record(
            name, call_args, request_id=request_id, session_id=session_id, ok=True
        )
        logger.info(f"guard 放行执行 {name} ({latency_ms:.0f}ms, 未注册={unregistered})")
        return GuardResult(ok=True, value=result)

    except Exception as e:  # noqa: BLE001 - 咽喉契约：失败折叠为结果对象
        _audit(name, call_args, "error", started, request_id, error=str(e))
        tool_trace_sink.record(
            name, call_args, request_id=request_id,
            session_id=session_id, ok=False, error=str(e),
        )
        logger.error(f"guard 执行 {name} 失败: {e}")
        return GuardResult(ok=False, error=str(e))


async def execute_approved(action: PendingAction) -> GuardResult:
    """人工 approve 后补执行：从待审动作取回工具与参数直接调用

    原子认领：先把状态从 approved 推进为 executed（条件 UPDATE，
    恰好一个并发方成功），认领失败即说明动作已被处理或状态不符，
    拒绝执行——重复/并发 approve 绝不会多次触发高风险操作。
    执行结果回填到动作记录。
    """
    store = get_pending_action_store()
    claimed = store.transition(action.action_id, ActionStatus.APPROVED, ActionStatus.EXECUTED)
    if claimed is None:
        return GuardResult(
            ok=False,
            error="动作不在 approved 状态（可能已被裁决或补执行），拒绝重放",
        )

    tool = await _find_tool(claimed.tool_name)
    if tool is None:
        message = f"失败: 找不到工具 {claimed.tool_name}，无法补执行"
        store.attach_result(action.action_id, message)
        return GuardResult(ok=False, error=message)

    result = await guarded_call(
        tool, claimed.args, request_id=claimed.request_id,
        session_id=claimed.session_id, approved_action_id=action.action_id,
    )
    # 补执行时确认门不再二次拦截（人已批准）；防御性处理意外状态
    if result.needs_confirmation:
        result = GuardResult(ok=False, error="补执行异常地再次要求确认，已中止")
    store.attach_result(action.action_id, result.value if result.ok else f"失败: {result.error}")
    return result


def decide_action(action_id: str, decision: ActionStatus) -> PendingAction | None:
    """裁决待审动作（approve/reject 端点共用）"""
    return get_pending_action_store().decide(action_id, decision)


# ──────────────── 内部 ────────────────


async def _propose_confirmation(
    name: str,
    args: dict[str, Any],
    *,
    reason: str,
    request_id: str,
    session_id: str,
) -> GuardResult:
    action: PendingAction = get_pending_action_store().propose(
        tool_name=name,
        args=args,
        reason=reason or f"工具 {name} 为高风险操作",
        session_id=session_id,
        request_id=request_id,
    )
    logger.warning(f"高风险工具 {name} 已挂起等待人工确认: action={action.action_id}")
    return GuardResult(
        ok=False,
        error=reason,
        needs_confirmation=True,
        action_id=action.action_id,
        meta={"risk_level": RiskLevel.WRITE_HIGH_RISK.value},
    )


def _denied(name: str, reason: str, started: float) -> GuardResult:
    _audit(name, {}, "rejected", started, "")
    return GuardResult(ok=False, error=reason)


async def _invoke_tool(tool: Any, args: dict[str, Any]) -> str:
    """统一经 langchain Runnable 协议执行（内部处理 sync/async 差异）"""
    result = await tool.ainvoke(args)
    if isinstance(result, str):
        return result
    return str(result)


_RISKY_KEY_PARTS = ("force", "confirm", "drop", "delete_all", "restart", "shutdown")
# 未注册工具参数值中的破坏性命令模式（保守集合，只拦明确意图）
_RISKY_VALUE_PATTERNS = (
    "rm -rf",
    "rm -fr",
    "drop table",
    "drop database",
    "delete from",
    "truncate table",
    "shutdown -h",
    "mkfs",
)


def _looks_destructive(args: Any) -> bool:
    """启发式：未注册工具的参数键名或字符串值携带破坏性意图即视为可疑

    递归扫描嵌套 dict/list（破坏载荷可能藏在 value 里而非顶层键）。
    这是纵深防御而非安全边界——真正的保障是「未注册工具先登记再执行」；
    此处只做保守拦截，正常只读 MCP 工具不受影响。
    """
    if isinstance(args, dict):
        for key, value in args.items():
            key_lower = str(key).lower()
            if any(part in key_lower for part in _RISKY_KEY_PARTS):
                return True
            if _looks_destructive(value):
                return True
    elif isinstance(args, (list, tuple)):
        return any(_looks_destructive(item) for item in args)
    elif isinstance(args, str):
        lowered = args.lower()
        return any(pattern in lowered for pattern in _RISKY_VALUE_PATTERNS)
    return False


def _audit(
    name: str,
    params: dict[str, Any],
    status: str,
    started: float,
    request_id: str,
    error: str | None = None,
) -> None:
    latency_ms = (time.perf_counter() - started) * 1000
    meta = tool_registry.get(name)
    # 已注册且未声明审计的工具不记（尊重注册表声明）；其余一律留痕
    if meta is not None and not meta.audit:
        return
    audit_logger.log_tool_call(
        request_id=request_id,
        tool_name=name,
        params=params,
        result_status=status,
        latency_ms=latency_ms,
        error=error,
    )


async def _find_tool(name: str) -> Any | None:
    """在本地与 MCP 工具池中按名查找（补执行路径用）"""
    from app.agent.runtime.toolsets import local_tool_map

    local = local_tool_map()
    if name in local:
        return local[name]
    try:
        from app.agent.mcp_client import get_mcp_tools

        for t in await get_mcp_tools():
            if getattr(t, "name", "") == name:
                return t
    except Exception as e:  # noqa: BLE001 - MCP 不可达时仍可回退本地查找结果
        logger.warning(f"补执行查找 MCP 工具失败: {e}")
    return None

