"""融合函数拆分回归（P5-b）：knowledge_tool 垫片必须指向 fusion 实现

机械迁移的防漂移锚：
- 别名身份：若有人在新代码里重写了同名私有函数，垫片别名与
  fusion 实现的身份相等性即被破坏，此处立刻报警。
- 无 IO 依赖（评审 #14 加固）：fusion 是纯函数模块，源码层面就
  不允许出现任何 app.services 导入——旧检查只看运行时命名空间，
  捕获不到「导入了但未绑定到易猜名字」的依赖。
"""

import inspect

import app.tools.knowledge_tool as kt
from app.services.retrieval import fusion
from app.services.retrieval.fusion import rrf_merge, rrf_merge_n, rrf_merge_three


class TestShimIdentity:
    def test_legacy_names_are_aliases_of_fusion_impl(self):
        assert kt._rrf_merge is rrf_merge
        assert kt._rrf_merge_n is rrf_merge_n
        assert kt._rrf_merge_three is rrf_merge_three


class TestPureFunctionBoundary:
    def test_fusion_source_has_no_app_service_imports(self):
        """源码文本扫描：fusion.py 不得导入任何 app.services / app.tools 模块"""
        source = inspect.getsource(fusion)
        for forbidden in ("from app.services", "import app.services", "from app.tools", "import app.tools"):
            assert forbidden not in source, f"fusion 源码出现禁止的依赖: {forbidden}"

    def test_knowledge_tool_calls_route_through_shim_aliases(self):
        """评审 #14：kt 内部对融合函数的调用必须全部走 _rrf_merge* 垫片别名——
        出现裸名直调即说明有人绕过了防漂移锚（别名被换掉时静默失联）"""
        import re

        source = inspect.getsource(kt)
        # 负向环视排除垫片别名自身（前导 _）与属性访问（fusion.rrf_merge）
        naked_calls = re.findall(
            r"(?<![A-Za-z0-9_.])rrf_merge(?:_n|_three)?\s*\(", source
        )
        assert naked_calls == []

    def test_rrf_merge_n_dedups_and_orders_by_reciprocal_rank(self, monkeypatch):
        from langchain_core.documents import Document

        merged = rrf_merge_n(
            [
                [Document(page_content="alpha"), Document(page_content="shared")],
                [Document(page_content="shared"), Document(page_content="beta")],
            ],
            k=60,
        )
        contents = [d.page_content for d in merged]
        # RRF 语义：双通道同时命中的 shared 累积两份倒数排名分居首，
        # 单命中文档按各自排名次之（去重保留首见位置）
        assert contents == ["shared", "alpha", "beta"]
