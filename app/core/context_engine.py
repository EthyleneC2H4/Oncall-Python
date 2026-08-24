"""上下文引擎 - 类型化 Packet + 分类型配额 + 预算内贪心装配 + 压缩

升级自 context_assembler（字符截断）与 token_budget（预算降级）：
统一为「Packet 模型」——任何注入 LLM 的上下文片段都是带类型与优先级的 Packet：

- MEMORY   记忆召回块（经验复用）
- KG       知识图谱分析
- RAG      检索文档段落
- HISTORY  对话历史
- MISC     其他（兜底类型）

装配算法（确定性、可做属性测试）：
1. 分类型配额填充：每类按自身配额（总预算 × 占比）内按优先级降序贪心选取
2. 剩余预算再分配：未入选 Packet 按全局优先级降序补位（单类超限 ≤ 1.5× 配额）
3. 压缩回收（可选）：高优先级但装不下的可压缩 Packet 经 compressor 缩短后重试；
   默认回退为「保留标记行」的确定性截断（[PLAN]/[结论]/[未解]/[doc] 行原样保留）

Token 计数复用 token_budget_manager 的中英混合估算（无 tiktoken 依赖）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger

from app.config import config
from app.core.token_budget import token_budget_manager

# 压缩时必须保留的语义标记前缀（roll-up 提示词与回退实现共用）
PRESERVE_MARKERS = ("[PLAN]", "[结论]", "[未解]", "[doc]")


class PacketKind(StrEnum):
    """上下文包类型"""

    MEMORY = "memory"
    KG = "kg"
    RAG = "rag"
    HISTORY = "history"
    MISC = "misc"


# 各类型的默认配额占比（总和 = 1.0；MISC 兜底类型也给最小配额，
# 否则其条目在阶段 1 永远装不下、只能靠再分配捡漏——形同虚设）
DEFAULT_QUOTA_FRACTIONS: dict[PacketKind, float] = {
    PacketKind.MEMORY: 0.20,
    PacketKind.KG: 0.25,
    PacketKind.RAG: 0.30,
    PacketKind.HISTORY: 0.20,
    PacketKind.MISC: 0.05,
}

# 各类型的默认优先级（数值越大越优先存活）
DEFAULT_KIND_PRIORITY: dict[PacketKind, int] = {
    PacketKind.MEMORY: 60,
    PacketKind.KG: 50,
    PacketKind.RAG: 40,
    PacketKind.HISTORY: 30,
    PacketKind.MISC: 10,
}

# 再分配阶段允许单类超出配额的倍数
SPILLOVER_FACTOR = 1.5


@dataclass
class Packet:
    """一个待注入 LLM 的上下文片段"""

    kind: PacketKind
    text: str
    priority: float | None = None  # None 时用类型默认优先级
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_priority(self) -> float:
        base = DEFAULT_KIND_PRIORITY.get(self.kind, 0)
        return float(base) if self.priority is None else self.priority

    @property
    def tokens(self) -> int:
        return token_budget_manager.estimate_tokens(self.text)


@dataclass
class AssembleResult:
    """装配结果：选中的包 + 渲染文本 + 统计"""

    selected: list[Packet]
    text: str
    stats: dict[str, Any]

    @property
    def used_tokens(self) -> int:
        return sum(p.tokens for p in self.selected)


Compressor = Callable[[str, PacketKind], Awaitable[str]]


class ContextEngine:
    """上下文引擎：在 Token 预算内装配最优上下文"""

    def __init__(
        self,
        *,
        quota_fractions: dict[PacketKind, float] | None = None,
        compressor: Compressor | None = None,
    ):
        """Args:
        quota_fractions: 分类型配额占比覆盖（按 key 合并进默认值，
            不整体替换——partial 覆盖不应把未提及类型的配额清零）
        compressor: 超预算高价值包的压缩器（如 weak-LLM roll-up）；
                    None 时用确定性回退截断
        """
        self.quota_fractions = {**DEFAULT_QUOTA_FRACTIONS, **(quota_fractions or {})}
        self.compressor = compressor

    # ──────────────── 主入口 ────────────────

    async def assemble(
        self,
        packets: list[Packet],
        *,
        budget: int | None = None,
        compress: bool = True,
    ) -> AssembleResult:
        """在预算内贪心装配；保证结果总 token ≤ budget（属性不变量）

        Args:
            packets: 待装配的上下文包（同一 Packet 实例重复传入只计一次）
            budget: 总 token 预算；默认 config.context_token_budget
            compress: 是否对装不下的高价值包执行压缩回收

        统计恒等式：input_packets == selected_count + dropped_count
        （compressed 以压缩形态计入 selected，其原始包不计入 dropped）
        """
        total_budget = budget if budget is not None else config.context_token_budget

        # 入口去重：簿记以 id(packet) 为键。同一实例的别名若重复计入，
        # 渲染会出现两份文本而预算只记一份，破坏 ≤ budget 不变量。
        nonempty: list[Packet] = []
        seen: set[int] = set()
        for p in packets:
            if p.text.strip() and id(p) not in seen:
                seen.add(id(p))
                nonempty.append(p)

        used = 0
        selected_ids: set[int] = set()
        per_kind_used: dict[PacketKind, int] = {}

        def _kind_cap(kind: PacketKind) -> int:
            fraction = self.quota_fractions.get(kind, 0.0)
            return int(total_budget * fraction * SPILLOVER_FACTOR)

        # 阶段 1：分类型配额填充（各类内部按优先级降序；零成本包免费入选）
        for kind in self.quota_fractions:
            quota = int(total_budget * self.quota_fractions.get(kind, 0.0))
            kind_budget = min(quota, max(total_budget - used, 0))
            if kind_budget <= 0:
                continue
            group = sorted(
                (p for p in nonempty if p.kind is kind),
                key=lambda p: p.effective_priority,
                reverse=True,
            )
            for packet in group:
                cost = packet.tokens
                if cost <= kind_budget:
                    selected_ids.add(id(packet))
                    kind_budget -= cost
                    used += cost
                    per_kind_used[kind] = per_kind_used.get(kind, 0) + cost

        # 阶段 2：剩余预算全局再分配（尊重 1.5× 配额上限）
        leftovers = sorted(
            (p for p in nonempty if id(p) not in selected_ids),
            key=lambda p: p.effective_priority,
            reverse=True,
        )
        for packet in leftovers:
            cost = packet.tokens
            if cost > total_budget - used:
                continue
            if cost > 0:
                kind_used_total = per_kind_used.get(packet.kind, 0)
                if kind_used_total + cost > _kind_cap(packet.kind):
                    continue
            selected_ids.add(id(packet))
            used += cost
            per_kind_used[packet.kind] = per_kind_used.get(packet.kind, 0) + cost

        # 阶段 3：压缩回收 —— 高优先级落选包压缩后重试装入
        compressed_packets: list[Packet] = []
        if compress and self._compressible(leftovers):
            for packet in leftovers:
                if id(packet) in selected_ids or not self._compressible([packet]):
                    continue
                slack = total_budget - used
                if packet.tokens <= slack:
                    continue  # 本来就装得下（理论上不会，防御）
                try:
                    shrunk_text = await self._compress(packet)
                except Exception as e:  # noqa: BLE001 - 压缩失败退回原文
                    logger.warning(f"上下文压缩失败（保留原文）: {e}")
                    continue
                shrunk = Packet(
                    kind=packet.kind,
                    text=shrunk_text,
                    priority=packet.effective_priority,
                    metadata={**packet.metadata, "compressed": True, "_origin": packet},
                )
                if shrunk.tokens > 0 and shrunk.tokens <= total_budget - used:
                    kind_used = per_kind_used.get(shrunk.kind, 0)
                    # 压缩形态同样受单类上限约束（否则可借压缩绕过 1.5× 配额）
                    if kind_used + shrunk.tokens > _kind_cap(shrunk.kind):
                        continue
                    compressed_packets.append(shrunk)
                    used += shrunk.tokens
                    per_kind_used[shrunk.kind] = kind_used + shrunk.tokens

        # 丢弃数 = 原始包中既未直接入选、也未以压缩形态装入的条数；
        # 口径与 input_packets 一致（都基于去重后的 nonempty），保证恒等式成立
        compressed_count = len(compressed_packets)
        survived_origins = {id(c.metadata.get("_origin")) for c in compressed_packets}
        dropped = sum(
            1
            for p in nonempty
            if id(p) not in selected_ids and id(p) not in survived_origins
        )

        selected = [p for p in nonempty if id(p) in selected_ids] + compressed_packets
        selected.sort(key=lambda p: (-p.effective_priority, p.kind.value))
        text = self.render(selected)

        stats = {
            "budget": total_budget,
            "used_tokens": used,
            "per_kind_tokens": {k.value: v for k, v in per_kind_used.items()},
            # 口径为去重后的有效输入（空白包不参与装配也不计入统计）
            "input_packets": len(nonempty),
            "selected_count": len(selected),
            "compressed_count": compressed_count,
            "dropped_count": dropped,
        }
        if dropped or compressed_count:
            logger.info(
                f"上下文装配: 预算 {total_budget} / 已用 {used}，"
                f"选中 {len(selected)}，压缩 {compressed_count}，丢弃 {dropped}"
            )
        return AssembleResult(selected=selected, text=text, stats=stats)

    # ──────────────── 渲染 ────────────────

    @staticmethod
    def render(selected: list[Packet]) -> str:
        """把选中的包渲染为注入文本（带类型小节标题）"""
        sections: list[str] = []
        order = [*PacketKind]
        for kind in order:
            group = [p for p in selected if p.kind is kind]
            if not group:
                continue
            header = {
                PacketKind.MEMORY: "[相关记忆]",
                PacketKind.KG: "[知识图谱]",
                PacketKind.RAG: "[参考文档]",
                PacketKind.HISTORY: "[对话历史]",
                PacketKind.MISC: "[补充信息]",
            }[kind]
            body = "\n".join(p.text.strip() for p in group)
            sections.append(f"{header}\n{body}")
        return "\n\n".join(sections)

    # ──────────────── 压缩 ────────────────

    def _compressible(self, packets: list[Packet]) -> bool:
        """仅 RAG / HISTORY / MEMORY 视为可压缩（KG 结构信息截断易失真）"""
        return any(p.kind in (PacketKind.RAG, PacketKind.HISTORY, PacketKind.MEMORY) for p in packets)

    async def _compress(self, packet: Packet) -> str:
        if self.compressor is not None:
            return await self.compressor(packet.text, packet.kind)
        return compress_fallback(packet.text)


def compress_fallback(text: str) -> str:
    """确定性回退压缩：保留标记行原样，其余内容截半

    保证无 LLM / LLM 失败时压缩路径依然可用且行为确定（可精确单测）。
    """
    lines = text.splitlines()
    kept_markers: list[str] = []
    others: list[str] = []
    for line in lines:
        if any(line.lstrip().startswith(marker) for marker in PRESERVE_MARKERS):
            kept_markers.append(line)
        else:
            others.append(line)
    other_text = "\n".join(others)
    half = max(len(other_text) // 2, 0)
    parts = ["\n".join(kept_markers)] if kept_markers else []
    if other_text:
        parts.append(other_text[:half] + ("\n…(已压缩)" if half < len(other_text) else ""))
    return "\n".join(parts).strip()


def make_weak_llm_compressor() -> Compressor:
    """构造 weak 层 LLM roll-up 压缩器（失败自动回退确定性截断）"""

    async def _compress(text: str, kind: PacketKind) -> str:
        from app.agent.runtime.llm_factory import tiered_llm

        llm = tiered_llm.weak()
        prompt = (
            "将以下运维上下文压缩到一半长度。要求：\n"
            f"- 以 {'/'.join(PRESERVE_MARKERS)} 开头的行必须逐字保留\n"
            "- 保留具体数字、服务名、错误关键字\n"
            "- 只输出压缩后的文本\n\n" + text
        )
        response = await llm.ainvoke(prompt)
        result = getattr(response, "content", "")
        if not isinstance(result, str) or len(result.strip()) < max(len(text) // 8, 8):
            raise ValueError("压缩输出过短或非文本，判定失败")
        return result.strip()

    async def _safe_compress(text: str, kind: PacketKind) -> str:
        try:
            return await _compress(text, kind)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"weak-LLM 压缩失败，使用确定性回退: {e}")
            return compress_fallback(text)

    return _safe_compress


def format_memory_block(items: list[Any], *, max_chars_per_item: int = 300) -> str:
    """把召回的记忆条目格式化为注入块（供运行时拼入 system prompt 尾部）

    Args:
        items: MemoryItem 列表（app.services.memory.types.MemoryItem）
        max_chars_per_item: 单条内容截断长度
    """
    if not items:
        return ""
    lines = []
    for item in items:
        preview = item.content.replace("\n", " ").strip()[:max_chars_per_item]
        type_label = item.type.value if hasattr(item.type, "value") else str(item.type)
        importance = getattr(item, "importance", 0.0)
        lines.append(f"- ({type_label}, 重要度 {importance:.1f}) {preview}")
    return "\n".join(lines)


# 全局单例（默认确定性回退压缩；需要 LLM roll-up 时用 make_weak_llm_compressor() 注入）
context_engine = ContextEngine()
