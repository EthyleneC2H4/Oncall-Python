"""结构化计划模型与容错解析

Planner 的 LLM 输出不可尽信（围栏包裹、截断、单引号伪 JSON、纯文本步骤……）。
parse_plan 提供一条「永不抛异常」的解析阶梯，保证任何输入都能退化为
可执行的 StructuredPlan —— 旧的 List[str] 计划行为永远可表示：

    passthrough → dict/steps → 围栏 JSON → 括号扫描 → 截断抢救 → 行模式兜底
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

# 截断抢救的最大回溯尝试次数（防超长垃圾输入的 O(n²) 扫描）
_MAX_SALVAGE_TRIES = 200


class PlanStep(BaseModel):
    """计划中的一个可执行步骤"""

    id: str = ""  # 空时由 StructuredPlan 归一化补齐序号
    description: str = Field(default="", description="步骤目标描述（自然语言）")
    tool: str | None = Field(default=None, description="绑定的工具名；None=由 Executor 自行决定")
    args: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")
    depends_on: list[str] = Field(default_factory=list, description="前置步骤 id 列表")
    expected_evidence: str | None = Field(
        default=None, description="本步骤预期产出的证据类型（日志/指标/文档…）"
    )


class StructuredPlan(BaseModel):
    """结构化执行计划"""

    steps: list[PlanStep] = Field(default_factory=list)
    # 解析来源（观测用途）：structured / fenced_json / truncated_json / lines / empty
    source_format: str = "structured"

    def __init__(self, **data: Any):
        super().__init__(**data)
        # 空描述步骤无执行意义（多为截断抢救的残渣），归一化时剔除；
        # 同步清理由此产生的悬挂 depends_on（以剔除前的原始 id 为准）
        kept_ids = {s.id for s in self.steps if s.id and s.description.strip()}
        self.steps = [s for s in self.steps if s.description.strip()]
        for step in self.steps:
            step.depends_on = [d for d in step.depends_on if d in kept_ids]
        self._normalize_ids()

    def _normalize_ids(self) -> None:
        """两遍扫描归一化 id

        第一遍裁定保留的显式 id（同 id 首见者胜）收入占用集合；
        第二遍对其余步骤从自身序号起找最小未占用正整数，
        避免盲取位置号与显式 id 或彼此撞车。
        """
        used: set[str] = set()
        reserved: list[bool] = []
        for step in self.steps:
            if step.id and step.id not in used:
                used.add(step.id)
                reserved.append(True)
            else:
                reserved.append(False)
        for i, (step, won) in enumerate(zip(self.steps, reserved, strict=True), 1):
            if won:
                continue
            candidate = i
            while str(candidate) in used:
                candidate += 1
            step.id = str(candidate)
            used.add(step.id)

    @property
    def legacy_strings(self) -> list[str]:
        """旧 List[str] 契约的等价表示（兼容 plan 事件/executor 既有消费方）"""
        return [step.description for step in self.steps]


def looks_like_plan(raw: Any) -> bool:
    """粗判：raw 是否承载了计划信息（即 parse_plan 有意义的处理对象）

    planner 层用它区分「结构化失败」（异常对象等）与「可容错解析的输出」：
    前者应走回退/默认计划，后者才交给 parse_plan 容错阶梯。
    """
    if raw is None:
        return False
    if isinstance(raw, (StructuredPlan, dict, list, str)):
        return True
    # 带 steps 属性的计划对象（旧 pydantic Plan / provider 结构化模型）
    return not isinstance(raw, (str, bytes, dict, list, tuple)) and hasattr(raw, "steps")


def parse_plan(raw: Any) -> StructuredPlan:
    """把任意 LLM 输出解析为 StructuredPlan（总函数：绝不抛异常）

    解析阶梯：
    1. 已是 StructuredPlan / PlanStep 列表 → 直接采用
    2. dict 含 steps → 递归其值
    3. 字符串 → 剥围栏 → 整体 JSON → 括号扫描提取 → 截断抢救 → ast.literal_eval
    4. 全部失败 → 行模式：逐行剥编号前缀作为无参步骤
    """
    try:
        return _parse(raw)
    except Exception as e:  # noqa: BLE001 - 总函数契约：解析失败不炸主流程
        logger.warning(f"parse_plan 兜底触发（{e}），退化为行模式")
        try:
            text = str(raw)
        except Exception:  # noqa: BLE001 - 连 __str__ 都炸的对象也要兜住
            text = f"<不可字符串化对象 {type(raw).__name__}>"
        return _plan_from_lines(text)


def _parse(raw: Any) -> StructuredPlan:
    if raw is None:
        return StructuredPlan(steps=[], source_format="empty")

    if isinstance(raw, StructuredPlan):
        return raw

    if isinstance(raw, PlanStep):
        return StructuredPlan(steps=[raw])

    # 泛型鸭子类型：任意带 steps 属性的计划对象（不同 provider 的
    # pydantic 返回模型、旧 Plan 类等）按其 steps 值递归解析
    if not isinstance(raw, (str, bytes, dict, list, tuple)) and hasattr(raw, "steps"):
        return _parse(raw.steps)

    if isinstance(raw, dict):
        if "steps" in raw:
            return _parse(raw["steps"])
        if "description" in raw:
            return StructuredPlan(steps=[PlanStep(**_safe_step_fields(raw))])
        # 未知形状的 dict：序列化后走行模式（至少保住信息）
        return _plan_from_lines(json.dumps(raw, ensure_ascii=False))

    if isinstance(raw, (list, tuple)):
        steps: list[PlanStep] = []
        for element in raw:
            if isinstance(element, PlanStep):
                steps.append(element)
            elif isinstance(element, dict):
                steps.append(PlanStep(**_safe_step_fields(element)))
            elif isinstance(element, str) and element.strip():
                steps.append(PlanStep(description=element.strip()))
        return StructuredPlan(steps=steps, source_format="structured")

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return StructuredPlan(steps=[], source_format="empty")

        # ① 剥代码围栏（```json ... ``` / ``` ... ```）
        fenced = _strip_code_fence(text)
        candidates = [fenced] if fenced else []
        candidates.append(text)

        for candidate in candidates:
            # ② 整体 JSON
            parsed = _try_loads(candidate)
            if parsed is not None:
                plan = _parse(parsed)
                plan.source_format = "fenced_json" if fenced else "structured"
                return plan

            # ③ 括号扫描：从文本中提取首个平衡的 {...} / [...]
            extracted = _extract_balanced(candidate)
            if extracted:
                parsed = _try_loads(extracted)
                if parsed is not None:
                    plan = _parse(parsed)
                    plan.source_format = "fenced_json"
                    return plan

            # ④ 截断抢救：JSON 被截断时从尾部回溯到最后一个可闭合的位置
            salvaged = _salvage_truncated(candidate)
            if salvaged is not None:
                plan = _parse(salvaged)
                plan.source_format = "truncated_json"
                return plan

            # ⑤ Python repr 风格（单引号伪 JSON）
            py_evaluated = _try_literal_eval(candidate)
            if py_evaluated is not None and isinstance(py_evaluated, (list, dict)):
                plan = _parse(py_evaluated)
                plan.source_format = "fenced_json"
                return plan

        # ⑥ 行模式兜底：旧 List[str] 行为永远可表示
        return _plan_from_lines(text)

    # 其余标量（int/float…）：单步无参计划
    return _plan_from_lines(str(raw))


def _plan_from_lines(text: str) -> StructuredPlan:
    """行模式兜底：每行一个无参步骤，剥掉常见编号/列表前缀"""
    # 注意：字符类中的全角冒号（U+FF1A）/全角括号（U+FF09）是字面字符，
    # 与 ASCII 版肉眼难辨——曾因误写成 ASCII 冒号导致「步骤2：」前缀漏剥。
    # (?!\.\d) 排除小数衔接，防止 "2.4GHz" 被截头成 "4GHz"
    prefix_re = re.compile(
        r"^\s*(?:步骤\s*\d+(?!\.\d)\s*[:：.]|\d+(?!\.\d)\s*[:：.、)）]|[-*•])\s*"
    )
    steps = []
    for line in text.splitlines():
        cleaned = prefix_re.sub("", line.strip())
        if cleaned:
            steps.append(PlanStep(description=cleaned))
    source = "lines" if steps else "empty"
    return StructuredPlan(steps=steps, source_format=source)


def _safe_step_fields(data: dict[str, Any]) -> dict[str, Any]:
    """从 dict 提取 PlanStep 已知字段（多余字段忽略，缺省字段补默认）"""
    known = {"id", "description", "tool", "args", "depends_on", "expected_evidence"}
    fields = {k: v for k, v in data.items() if k in known}
    if "args" in fields and not isinstance(fields["args"], dict):
        fields["args"] = {"value": fields["args"]}
    if "depends_on" in fields and isinstance(fields["depends_on"], str):
        fields["depends_on"] = [fields["depends_on"]]
    return fields


def _strip_code_fence(text: str) -> str | None:
    """剥 ```json …``` / ``` …``` 围栏；无围栏返回 None"""
    fence_re = re.compile(r"^```[a-zA-Z0-9]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
    match = fence_re.match(text)
    return match.group(1).strip() if match else None


def _try_loads(text: str) -> Any | None:
    """json.loads 容错包装；失败返回 None"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_balanced(text: str) -> str | None:
    """括号扫描：提取首个顶层平衡的 {...} 或 [...] 片段（字符串字面量内不计数）"""
    opener = next((ch for ch in text if ch in "{["), None)
    if opener is None:
        return None
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False
    start = text.index(opener)
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and ch == closer:
                return text[start : i + 1]
    return None


def _salvage_truncated(text: str) -> Any | None:
    """截断 JSON 抢救：从尾部回溯，找到最后一个能闭合的完整元素处截断加载

    场景：max_tokens 截断输出 `"steps": [{"description": "查日志"}, {"desc` ——
    抢救出第一个完整元素而非整体失败。
    """
    positions = [
        i for i, ch in enumerate(text) if ch in "}]" and i < len(text) - 1
    ]
    tries = 0
    for pos in reversed(positions):
        tries += 1
        if tries > _MAX_SALVAGE_TRIES:
            return None
        candidate = text[: pos + 1]
        # 补齐缺失的收口括号再试
        balanced = _close_brackets(candidate)
        parsed = _try_loads(balanced)
        if parsed is not None:
            return parsed
    return None


def _close_brackets(text: str) -> str:
    """为未闭合的括号/引号补齐收尾（仅用于截断抢救尝试）"""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    suffix = "".join("}" if c == "{" else "]" for c in reversed(stack))
    return text + suffix


def _try_literal_eval(text: str) -> Any | None:
    """ast.literal_eval 容错包装（单引号伪 JSON / Python repr）；失败返回 None"""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def plan_to_legacy_strings(plan: StructuredPlan | list[str] | Any) -> list[str]:
    """任意计划形态 → 旧 List[str] 契约（事件兼容层使用）"""
    if isinstance(plan, StructuredPlan):
        return plan.legacy_strings
    if isinstance(plan, list):
        return [str(item) for item in plan]
    return parse_plan(plan).legacy_strings
