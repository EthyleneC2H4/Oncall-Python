"""Prompt 可组合块 + AB 变体测试（P5-a）

覆盖：块库加载与无效块跳过、render_composed 组装顺序、变体解析与
回退、intent 驱动的 few-shot 选择、热加载，以及真实 prompts/ 目录
的组合集成。
"""


import pytest
import yaml

from app.core.cost_tracker import CostTracker
from app.core.prompt_manager import PromptManager

TEMPLATE_YAML = """
name: sys_test
version: 1
description: 测试模板
variables:
  - {name: topic, required: false}
blocks:
  - persona_t
  - rules_t
variants:
  tight:
    description: 紧凑变体
    content: |
      紧凑正文：{topic}
content: |
  基线正文：{topic}
"""

PERSONA_YAML = """
name: persona_t
kind: persona
intents: []
priority: 10
description: 角色块
content: |
  你是测试角色。
"""

RULES_YAML = """
name: rules_t
kind: rules
intents: []
priority: 10
description: 规则块
content: |
  不要编造。
"""

FEWSHOT_DIAG_YAML = """
name: fs_diag
kind: few_shot
intents: [DIAGNOSTIC]
priority: 10
description: 诊断示例
content: |
  示例：诊断流程……
"""

FEWSHOT_CHAT_YAML = """
name: fs_chat
kind: few_shot
intents: [CHITCHAT]
priority: 20
description: 闲聊示例
content: |
  示例：礼貌引导……
"""


def _write(path, text):
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def pm(tmp_path):
    _write(tmp_path / "sys.yaml", TEMPLATE_YAML)
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    _write(blocks / "persona.yaml", PERSONA_YAML)
    _write(blocks / "rules.yaml", RULES_YAML)
    _write(blocks / "fs_diag.yaml", FEWSHOT_DIAG_YAML)
    _write(blocks / "fs_chat.yaml", FEWSHOT_CHAT_YAML)
    return PromptManager(prompts_dir=str(tmp_path))


class TestBlockLoading:
    def test_blocks_loaded_from_subdir(self, pm):
        names = {b["name"] for b in pm.list_blocks()}
        assert {"persona_t", "rules_t", "fs_diag", "fs_chat"} <= names

    def test_template_exposes_blocks_and_variants(self, pm):
        t = pm.get("sys_test")
        assert t.blocks == ["persona_t", "rules_t"]
        assert t.variant_names() == ["tight"]
        assert pm.list_templates()[0]["variants"] == ["tight"]


class TestRenderComposed:
    def test_order_persona_body_rules(self, pm):
        out = pm.render_composed("sys_test")
        assert out.index("你是测试角色") < out.index("基线正文") < out.index("不要编造")

    def test_variables_substituted_across_composition(self, pm):
        assert "基线正文：CPU" in pm.render_composed("sys_test", topic="CPU")

    def test_variant_replaces_body_keeps_blocks(self, pm):
        out = pm.render_composed("sys_test", variant="tight", topic="内存")
        assert "紧凑正文：内存" in out
        assert "你是测试角色" in out  # persona 块不受变体影响
        assert "基线正文" not in out

    def test_unknown_variant_falls_back_to_base(self, pm):
        out = pm.render_composed("sys_test", variant="ghost", topic="x")
        assert "基线正文：x" in out
        assert pm.effective_variant("sys_test", "ghost") == ""

    def test_effective_variant_resolution(self, pm):
        assert pm.effective_variant("sys_test", None) == ""
        assert pm.effective_variant("sys_test", "tight") == "tight"


class TestIntentFewShots:
    def test_diagnostic_intent_pulls_only_diag_fewshot(self, pm):
        out = pm.render_composed("sys_test", intent="DIAGNOSTIC")
        assert "诊断流程" in out
        assert "礼貌引导" not in out

    def test_chitchat_intent_pulls_only_chat_fewshot(self, pm):
        out = pm.render_composed("sys_test", intent="CHITCHAT")
        assert "礼貌引导" in out and "诊断流程" not in out

    def test_no_intent_injects_no_fewshot(self, pm):
        out = pm.render_composed("sys_test")
        assert "示例：" not in out


class TestHotReload:
    def test_block_file_change_picked_up(self, pm, tmp_path):
        before = pm.get_block("rules_t").content
        block_file = tmp_path / "blocks" / "rules.yaml"
        data = yaml.safe_load(RULES_YAML)
        data["content"] = "新规则内容。\n"
        _write(block_file, yaml.safe_dump(data, allow_unicode=True))
        # mtime 精度不足时强制前进
        st = block_file.stat()
        import os

        os.utime(block_file, (st.st_atime + 2, st.st_mtime + 2))

        after = pm.get_block("rules_t").content
        assert before != after
        assert "新规则内容" in pm.render_composed("sys_test")

    def test_invalid_block_skipped_not_fatal(self, tmp_path):
        _write(tmp_path / "sys.yaml", TEMPLATE_YAML)
        blocks = tmp_path / "blocks"
        blocks.mkdir()
        _write(blocks / "bad_kind.yaml", "name: bad\nkind: unknown_kind\ncontent: x\n")
        _write(blocks / "no_name.yaml", "kind: rules\ncontent: y\n")
        pm = PromptManager(prompts_dir=str(tmp_path))
        assert pm.get_block("bad") is None  # kind 越界 → 拒载
        assert pm.get_block("rules") is None  # 缺 name（无名块不入库）
        assert list(pm.list_blocks()) == []  # 两个坏块全部被跳过
        composed = pm.render_composed("sys_test")
        assert "基线正文" in composed  # 模板组合不受坏块影响


class TestRealRepoIntegration:
    def test_repo_system_prompt_composes_persona_and_rules(self):
        """真实 prompts/ 目录：system_prompt 组合后包含角色与硬约束"""
        from app.core.prompt_manager import prompt_manager as repo_pm

        composed = repo_pm.render_composed("system_prompt")
        assert "AIOps Agent" in composed  # persona_oncall 块
        assert "不要编造" in composed  # rules_grounding 块
        assert "执行流" in composed  # 模板正文
        assert "示例：" not in composed  # 未给意图时不注入 few-shot

    def test_repo_concise_variant_differs_from_base(self):
        from app.core.prompt_manager import prompt_manager as repo_pm

        base = repo_pm.render_composed("system_prompt")
        tight = repo_pm.render_composed("system_prompt", variant="concise")
        assert base != tight
        assert repo_pm.effective_variant("system_prompt", "concise") == "concise"


class TestVariantAttribution:
    def test_mark_prompt_variant_groups_sessions(self):
        tracker = CostTracker()
        tracker.mark_prompt_variant("", session_id="s1")
        tracker.mark_prompt_variant("concise", session_id="s1")
        tracker.mark_prompt_variant("concise", session_id="s1")  # 同会话重复
        tracker.mark_prompt_variant("concise", session_id="s2")

        summary = tracker.get_summary()
        assert summary["prompt_variants"]["base"]["runs"] == 1
        assert summary["prompt_variants"]["concise"]["runs"] == 3
        assert summary["prompt_variants"]["concise"]["sessions"] == 2
