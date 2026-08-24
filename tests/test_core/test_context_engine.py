"""ContextEngine 单测：预算不变量（随机洪水）/ 配额 / 压缩 / 渲染 / 记忆块格式化"""

import random

from app.core.context_engine import (
    ContextEngine,
    Packet,
    PacketKind,
    compress_fallback,
    format_memory_block,
    make_weak_llm_compressor,
)
from app.services.memory.types import MemoryItem, MemoryType


class _TieredLLMStub:
    """tiered_llm 替身（供 weak-LLM 压缩器回退测试注入）"""

    def __init__(self, weak_factory):
        self._weak_factory = weak_factory

    def weak(self):
        return self._weak_factory()


def _flood_packets(rng: random.Random, n: int = 120) -> list[Packet]:
    """生成随机洪水：类型/长度/优先级全随机"""
    kinds = list(PacketKind)
    packets = []
    for _ in range(n):
        kind = rng.choice(kinds)
        size = rng.choice([10, 50, 200, 1000, 5000])  # 字符量级跨度大
        text = rng.choice(["上下", "word "]) * max(size // 2, 1)
        priority = rng.uniform(0, 100)
        packets.append(Packet(kind=kind, text=text, priority=priority))
    return packets


class TestBudgetInvariant:
    async def test_random_flood_never_exceeds_budget(self):
        """属性测试：任意洪水输入下，装配结果总 token ≤ 预算"""
        rng = random.Random(42)
        for trial in range(20):
            budget = rng.randint(300, 4000)
            engine = ContextEngine()
            result = await engine.assemble(_flood_packets(rng), budget=budget, compress=False)
            assert result.used_tokens <= budget, f"trial={trial} 超预算: {result.used_tokens} > {budget}"

    async def test_alias_duplicates_never_exceed_budget(self):
        """回归 #10：同一 Packet 实例重复传入（别名）不得破坏预算不变量。

        簿记以 id() 为键，若不去重，渲染会出现两份文本而预算只记一份。
        """
        base = _flood_packets(random.Random(7), n=40)
        aliased = base + base[:20] + [base[0]] * 5  # 大量重复别名
        engine = ContextEngine()
        result = await engine.assemble(aliased, budget=800, compress=False)
        assert result.used_tokens <= 800
        # 去重后统计口径：input == selected + dropped
        assert result.stats["input_packets"] == len(base)
        assert (
            result.stats["input_packets"]
            == result.stats["selected_count"] + result.stats["dropped_count"]
        )
        # 渲染文本中无重复条目
        assert result.selected.count(base[0]) <= 1

    async def test_stats_identity_holds_under_compression(self):
        """恒等式 input == selected + dropped 在压缩路径同样成立"""
        rng = random.Random(99)

        async def compressor(text: str, kind: PacketKind) -> str:
            return "短"

        engine = ContextEngine(compressor=compressor)
        result = await engine.assemble(
            _flood_packets(rng), budget=1500, compress=True
        )
        s = result.stats
        assert s["input_packets"] == s["selected_count"] + s["dropped_count"]
        assert result.used_tokens <= 1500

    async def test_zero_cost_packet_not_counted_as_dropped(self):
        """回归 #13：零成本包免费入选，不再被误计为丢弃"""
        engine = ContextEngine()
        tiny = Packet(kind=PacketKind.MISC, text="x")
        result = await engine.assemble([tiny], budget=50, compress=False)
        assert result.stats["selected_count"] == 1
        assert result.stats["dropped_count"] == 0

    async def test_whitespace_only_packets_excluded_from_stats(self):
        """空白包不参与装配也不计入任何统计口径"""
        engine = ContextEngine()
        result = await engine.assemble(
            [Packet(kind=PacketKind.MISC, text="   \n")], budget=500
        )
        assert result.stats["input_packets"] == 0
        assert result.selected == []

    async def test_empty_input(self):
        engine = ContextEngine()
        result = await engine.assemble([], budget=1000)
        assert result.selected == []
        assert result.text == ""
        assert result.used_tokens == 0

    async def test_single_small_packet_always_selected(self):
        engine = ContextEngine()
        packet = Packet(kind=PacketKind.KG, text="小片段")
        result = await engine.assemble([packet], budget=10_000)
        assert [p.text for p in result.selected] == ["小片段"]


class TestQuotaAndPriority:
    async def test_high_priority_survives_tight_budget(self):
        """预算紧张时，高优先级小包应胜过低优先级大包"""
        engine = ContextEngine()
        big_low = Packet(kind=PacketKind.RAG, text="低" * 8000, priority=1.0)
        small_high = Packet(kind=PacketKind.MEMORY, text="关键记忆", priority=99.0)
        result = await engine.assemble(
            [big_low, small_high], budget=300, compress=False
        )
        texts = [p.text for p in result.selected]
        assert "关键记忆" in texts
        assert ("低" * 8000) not in texts

    async def test_spillover_cap_respected(self):
        """再分配阶段单类不得超过 1.5× 配额"""
        engine = ContextEngine(quota_fractions={PacketKind.RAG: 0.5, PacketKind.MISC: 0.0})
        packets = [
            Packet(kind=PacketKind.RAG, text=f"rag-{i}-" + "内容" * 40, priority=float(i))
            for i in range(10)
        ]
        budget = 1000
        result = await engine.assemble(packets, budget=budget, compress=False)
        rag_used = result.stats["per_kind_tokens"].get("rag", 0)
        cap = int(budget * 0.5 * 1.5)
        assert rag_used <= cap

    async def test_partial_quota_override_keeps_defaults(self):
        """回归 #11：partial 覆盖按 key 合并，未提及类型的配额不被清零"""
        engine = ContextEngine(quota_fractions={PacketKind.KG: 0.6})
        # 未提及的 MEMORY 应保有默认 0.20 配额——大预算下记忆包能直接入选
        memory_packet = Packet(kind=PacketKind.MEMORY, text="记忆内容", priority=50.0)
        result = await engine.assemble([memory_packet], budget=10_000, compress=False)
        assert result.stats["per_kind_tokens"].get("memory", 0) > 0

    async def test_misc_kind_has_nonzero_default_quota(self):
        """回归 #11：MISC 默认配额非零（兜底类型不再形同虚设）"""
        from app.core.context_engine import DEFAULT_QUOTA_FRACTIONS

        assert DEFAULT_QUOTA_FRACTIONS[PacketKind.MISC] > 0.0
        engine = ContextEngine()
        misc = Packet(kind=PacketKind.MISC, text="补充" * 30, priority=99.0)
        result = await engine.assemble([misc], budget=2000, compress=False)
        assert result.stats["per_kind_tokens"].get("misc", 0) > 0

    async def test_stats_shape(self):
        engine = ContextEngine()
        result = await engine.assemble(
            [Packet(kind=PacketKind.KG, text="kg 内容")], budget=500
        )
        stats = result.stats
        assert stats["budget"] == 500
        assert stats["input_packets"] == 1
        assert stats["selected_count"] == 1
        assert stats["dropped_count"] == 0


class TestCompression:
    async def test_fallback_preserves_marker_lines_verbatim(self):
        text = "[PLAN] 第一步查日志\n普通叙述内容A\n[结论] 根因是内存泄漏\n普通叙述内容B\n[未解] 为何夜间高发"
        compressed = compress_fallback(text)
        for line in compressed.splitlines():
            if line.startswith(("[PLAN]", "[结论]", "[未解]")):
                assert line in text  # 标记行逐字保留

    async def test_fallback_shortens_long_text(self):
        text = "\n".join(f"填充行{i}" + "x" * 50 for i in range(20))
        compressed = compress_fallback(text)
        assert len(compressed) < len(text)
        assert "已压缩" in compressed

    async def test_compressed_packet_enters_result_and_counts(self):
        """压缩成功的包必须真实进入结果文本与 token 统计"""

        async def compressor(text: str, kind: PacketKind) -> str:
            return "压缩后的短文本"

        engine = ContextEngine(compressor=compressor)
        huge = Packet(kind=PacketKind.RAG, text="很长" * 2000, priority=50.0)
        result = await engine.assemble([huge], budget=100, compress=True)

        assert result.stats["compressed_count"] == 1
        assert any(p.metadata.get("compressed") for p in result.selected)
        assert "压缩后的短文本" in result.text
        assert result.used_tokens <= 100

    async def test_kg_kind_not_compressible(self):
        """KG 类型不参与压缩（结构信息截断易失真）——压缩失败即丢弃"""
        calls: list[str] = []

        async def compressor(text: str, kind: PacketKind) -> str:
            calls.append(kind.value)
            return "短"

        engine = ContextEngine(compressor=compressor)
        kg = Packet(kind=PacketKind.KG, text="图谱" * 500, priority=10.0)
        result = await engine.assemble([kg], budget=50, compress=True)
        assert calls == []  # 未被压缩
        assert result.stats["dropped_count"] == 1

    async def test_compression_respects_kind_cap(self):
        """回归 #12：压缩形态的装入同样受 1.5× 单类上限约束，不得借压缩绕过"""
        engine = ContextEngine(quota_fractions={PacketKind.RAG: 0.1})

        async def compressor(text: str, kind: PacketKind) -> str:
            return "压" * 30  # 压缩产物仍不小（≈60 tok）

        # 先用大量小 RAG 包把配额与再分配余量吃满
        packets = [
            Packet(kind=PacketKind.RAG, text=f"填充{i}" + "x" * 20, priority=float(i))
            for i in range(12)
        ]
        big = Packet(kind=PacketKind.RAG, text="超长文档" * 500, priority=-1.0)
        result = await engine.assemble(packets + [big], budget=300, compress=True)

        rag_used = result.stats["per_kind_tokens"].get("rag", 0)
        cap = int(300 * 0.1 * 1.5)  # 45
        assert rag_used <= cap, f"压缩后单类用量 {rag_used} 超上限 {cap}"
        assert result.used_tokens <= 300

    async def test_weak_llm_compressor_falls_back_on_error(self):
        """weak-LLM 压缩器在 LLM 失败时回退确定性截断而非抛异常"""
        compressor = make_weak_llm_compressor()

        class ExplodingLLM:
            async def ainvoke(self, prompt: str):
                raise RuntimeError("OpenRouter 超时")

        import app.agent.runtime.llm_factory as lf

        original = lf.tiered_llm
        try:
            lf.tiered_llm = _TieredLLMStub(weak_factory=lambda: ExplodingLLM())
            text = "正文" * 100 + "\n[结论] 保留我"
            out = await compressor(text, PacketKind.HISTORY)
            assert "[结论] 保留我" in out
            assert len(out) < len(text)
        finally:
            lf.tiered_llm = original


class TestRenderAndMemoryBlock:
    def test_render_groups_by_kind_with_headers(self):
        selected = [
            Packet(kind=PacketKind.RAG, text="文档段落"),
            Packet(kind=PacketKind.MEMORY, text="记忆条目"),
        ]
        text = ContextEngine.render(selected)
        assert "[相关记忆]" in text and "记忆条目" in text
        assert "[参考文档]" in text and "文档段落" in text
        # 记忆节排在文档节之前（优先级降序）
        assert text.index("[相关记忆]") < text.index("[参考文档]")

    def test_format_memory_block(self):
        items = [
            MemoryItem(type=MemoryType.SEMANTIC, content="历史事故经验", importance=0.7),
            MemoryItem(type=MemoryType.EPISODIC, content="上次诊断记录", importance=0.3),
        ]
        block = format_memory_block(items)
        assert block.count("\n- ") == 1  # 两条目各占一行
        assert "(semantic, 重要度 0.7)" in block
        assert "历史事故经验" in block

    def test_format_memory_block_truncates_long_content(self):
        item = MemoryItem(type=MemoryType.EPISODIC, content="长" * 1000)
        block = format_memory_block([item], max_chars_per_item=50)
        assert len(block) < 200

    def test_format_memory_block_empty(self):
        assert format_memory_block([]) == ""
