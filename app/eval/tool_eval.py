"""BFCL 式工具调用评测 —— 参数级 AST 匹配（纯函数，离线可跑）

数据源：guard 执行咽喉落盘的 data/traces/tools.jsonl（完整实参）。
金标：dataset_registry 版本化的 {query/scenario → expected_tool_calls}。

匹配语义（借鉴 BFCL / Berkeley Function-Calling Leaderboard）：
- 工具名必须精确相等
- 实参与形参做类型敏感比较——bool 与 int 是 Python 里 ==
  相等的不同类型（True == 1），运维参数里开关与数量混淆是真实缺陷，
  因此 bool 永远不匹配数字
- 列表默认按序；unordered_lists=True 时作多重集合比较
- 浮点在 float_tolerance 内视为相等（默认严格）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ToolCallVerdict:
    """单条工具调用的比对结论"""

    matched: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class TraceEntry:
    """tools.jsonl 中的一条工具调用记录"""

    tool_name: str
    args: dict[str, Any]
    request_id: str = ""
    session_id: str = ""
    ok: bool = True
    timestamp: float = 0.0


# ──────────────── 值级规范化比对 ────────────────


def canonical_value(value: Any) -> Any:
    """递归规范化：bytes→str、元组→列表（其余原样，类型语义保留）"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return [canonical_value(v) for v in value]
    if isinstance(value, list):
        return [canonical_value(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical_value(v) for k, v in value.items()}
    return value


def _values_match(expected: Any, actual: Any, *, tolerance: float | None) -> bool:
    """类型敏感的值相等判断（容器内递归保持同一语义）

    关键陷阱：Python `True == 1` 为真、`1 == 1.0` 为真，且容器 ==
    会用这套宽松语义逐元素比较（[True] == [1]、{"a": True} == {"a": 1}
    均为真）。这里要求「数值家族内可比，但 bool 独立」且递归到底：
      bool vs 非.bool → 不匹配（含嵌套在 list/dict 内）
      int/float 互相 → 数值相等即可
    """
    exp_t, act_t = type(expected), type(actual)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(exp_t is act_t and expected == actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if tolerance is not None:
            return abs(float(expected) - float(actual)) <= tolerance
        return bool(expected == actual)
    if exp_t is not act_t:
        # str vs 数字等跨类型一律不等（"80" ≠ 80，LLM 幻觉常见形状）
        return False
    if isinstance(expected, list):
        # 同型已保证 actual 是 list；逐元素递归而非 Python ==
        return bool(
            len(expected) == len(actual)
            and all(
                _values_match(e, a, tolerance=tolerance)
                for e, a in zip(expected, actual, strict=True)
            )
        )
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return False
        return bool(
            all(_values_match(v, actual[k], tolerance=tolerance) for k, v in expected.items())
        )
    return bool(expected == actual)


def match_arguments(
    expected_args: dict[str, Any],
    actual_args: dict[str, Any],
    *,
    unordered_lists: bool = False,
    float_tolerance: float | None = None,
) -> ToolCallVerdict:
    """比对该次调用的实参与期望参数（纯函数）

    规则：
    - 缺失期望键 / 多出未知键都算 mismatch 并写明原因
    - None 值与缺键等价（LLM 显式传 null ≈ 未提供）
    """
    expected = canonical_value(expected_args or {})
    actual_raw = dict(actual_args or {})
    # 显式 None 归一为「未提供」
    actual = {k: v for k, v in canonical_value(actual_raw).items() if v is not None}

    reasons: list[str] = []
    for key, exp in expected.items():
        if exp is None:
            continue
        if key not in actual:
            reasons.append(f"缺少参数 {key}（期望 {exp!r}）")
            continue
        act = actual[key]
        if unordered_lists and isinstance(exp, list) and isinstance(act, list):
            if sorted(map(_sort_key, exp)) != sorted(map(_sort_key, act)):
                reasons.append(f"参数 {key} 集合不符: 期望 {exp!r}, 实际 {act!r}")
        elif not _values_match(exp, act, tolerance=float_tolerance):
            reasons.append(
                f"参数 {key} 不符: 期望 {exp!r} ({type(exp).__name__}), "
                f"实际 {act!r} ({type(act).__name__})"
            )

    extra = sorted(set(actual) - {k for k in expected if expected[k] is not None})
    if extra:
        reasons.append(f"多出参数: {extra}")

    return ToolCallVerdict(matched=not reasons, reasons=reasons)


def _sort_key(value: Any) -> str:
    """列表无序比较用的稳定排序键（json 序列化兜底任意嵌套）"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def match_tool_call(
    expected: dict[str, Any],
    actual: TraceEntry,
    *,
    unordered_lists: bool = False,
    float_tolerance: float | None = None,
) -> ToolCallVerdict:
    """比对一次期望的工具调用与实际痕迹

    expected: {"tool": "search_log", "args": {...}}
    """
    exp_tool = expected.get("tool", "")
    if exp_tool != actual.tool_name:
        return ToolCallVerdict(
            matched=False, reasons=[f"工具名不符: 期望 {exp_tool!r}, 实际 {actual.tool_name!r}"]
        )
    verdict = match_arguments(
        expected.get("args") or {},
        actual.args,
        unordered_lists=unordered_lists,
        float_tolerance=float_tolerance,
    )
    if not verdict.matched:
        verdict.reasons.insert(0, f"工具 {exp_tool} 参数不匹配")
    return verdict


# ──────────────── 痕迹加载与会话聚合 ────────────────


def load_traces(path: str | Path) -> list[TraceEntry]:
    """加载 tools.jsonl（坏行跳过并警告，不中断回放）"""
    file_path = Path(path)
    if not file_path.exists():
        return []
    entries: list[TraceEntry] = []
    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entries.append(
                    TraceEntry(
                        tool_name=str(raw.get("tool_name", "")),
                        args=dict(raw.get("args") or {}),
                        request_id=str(raw.get("request_id", "")),
                        session_id=str(raw.get("session_id", "")),
                        ok=bool(raw.get("ok", True)),
                        timestamp=float(raw.get("timestamp", 0.0)),
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"[tool_eval] 跳过损坏行 {file_path.name}:{line_no}: {e}")
    return entries


@dataclass
class ScenarioScore:
    """一个场景（一组期望调用 vs 一段会话痕迹）的得分"""

    scenario_id: str
    total_expected: int
    matched: int
    details: list[dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.matched / self.total_expected if self.total_expected else 0.0


def evaluate_scenario(
    scenario_id: str,
    expected_calls: list[dict[str, Any]],
    trace_entries: list[TraceEntry],
    *,
    unordered_lists: bool = False,
    float_tolerance: float | None = None,
    allow_failures: bool = False,
) -> ScenarioScore:
    """贪心匹配：每条期望调用至多消费一条未使用的实际痕迹

    执行失败的痕迹（ok=False）默认不可被消费为匹配——「参数对了但
    调用炸了」不是正确的工具执行，放行会把门禁朝假绿方向抬高；
    allow_failures=True 时恢复只看参数的宽松口径。

    返回逐条明细（matched/reasons），供报告展示失败模式。
    """
    used: set[int] = set()
    details: list[dict] = []
    matched_count = 0
    saw_failed_candidate = False

    for idx, expected in enumerate(expected_calls):
        found_at: int | None = None
        verdict: ToolCallVerdict | None = None
        for pos, entry in enumerate(trace_entries):
            if pos in used:
                continue
            if not entry.ok and not allow_failures:
                saw_failed_candidate = True
                continue
            candidate = match_tool_call(
                expected,
                entry,
                unordered_lists=unordered_lists,
                float_tolerance=float_tolerance,
            )
            if candidate.matched:
                found_at = pos
                verdict = candidate
                break
            if verdict is None or len(candidate.reasons) < len(verdict.reasons):
                verdict = candidate  # 记录最接近的失败原因用于诊断
        if found_at is not None:
            used.add(found_at)
            matched_count += 1
            details.append({"index": idx, "matched": True, "tool": expected.get("tool")})
        else:
            if verdict is None and saw_failed_candidate:
                reasons = ["候选调用均执行失败(ok=False)，不计为匹配"]
            else:
                reasons = verdict.reasons if verdict else ["痕迹中无任何候选调用"]
            details.append({"index": idx, "matched": False, "reasons": reasons})

    return ScenarioScore(
        scenario_id=scenario_id,
        total_expected=len(expected_calls),
        matched=matched_count,
        details=details,
    )
