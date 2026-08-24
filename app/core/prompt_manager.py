"""Prompt 版本化管理 + 可组合块（P5）

功能：
- 从 YAML 文件加载 Prompt 模板（prompts/*.yaml）
- 从 prompts/blocks/ 加载可复用块（persona/rules/few_shot，按 intent 标签）
- 支持版本管理和变量校验
- render_composed()：块组合 + AB 变体解析后整体渲染
- 文件变更自动热加载（mtime 检查）
- 与 Trace 关联 prompt_name + prompt_version
"""

import os
import time
from pathlib import Path

import yaml
from loguru import logger

# 块种类 → 组合时相对模板正文的位置顺序（persona 在前、few_shot 收尾）
_KIND_ORDER = {"persona": 0, "rules": 2, "few_shot": 3}
_VALID_KINDS = set(_KIND_ORDER)


class PromptBlock:
    """可复用提示词块（prompts/blocks/*.yaml）

    Schema:
        name: str          全局唯一名
        kind: str          persona | rules | few_shot
        intents: [str]     适用的意图标签；空 = 通用块
        priority: int      同类多块时的排序权重（小者在前，默认 50）
        description: str   人类可读说明（不进提示词）
        content: str       块文本
    """

    def __init__(self, data: dict, file_path: str = ""):
        self.name: str = str(data.get("name", ""))
        self.kind: str = str(data.get("kind", ""))
        # intent 标签统一大写存储，匹配时对入参做同样归一（DIAGNOSTIC/diagnostic 等价）
        self.intents: list[str] = [str(i).upper() for i in (data.get("intents") or [])]
        self.priority: int = int(data.get("priority", 50))
        self.description: str = str(data.get("description", ""))
        self.content: str = str(data.get("content", ""))
        self.file_path = file_path
        self.loaded_at = time.time()

    @property
    def is_valid(self) -> bool:
        """缺 name/kind/content 或 kind 不在白名单的块不可用"""
        return bool(self.name and self.content) and self.kind in _VALID_KINDS

    def matches_intent(self, intent: str | None) -> bool:
        """意图匹配：通用块恒适用；带标签块仅在显式命中时适用

        intent=None 表示调用方未做意图分类——此时只保留通用块，
        不注入任何面向特定意图的 few-shot。
        （评审修复：入参归一大写，与标签存储口径一致）
        """
        if not self.intents:
            return True
        return intent is not None and str(intent).strip().upper() in self.intents


class PromptTemplate:
    """单个 Prompt 模板"""

    def __init__(self, data: dict, file_path: str = ""):
        self.name: str = data.get("name", "")
        self.version: str = str(data.get("version", "1"))
        self.description: str = data.get("description", "")
        self.variables: list[dict] = data.get("variables", [])
        self.model_config: dict = data.get("model_config", {})
        self.content: str = data.get("content", "")
        # P5 可组合扩展：声明的块名列表 + AB 变体表
        self.blocks: list[str] = [str(b) for b in (data.get("blocks") or [])]
        self.variants: dict[str, dict] = {
            str(k): dict(v or {}) for k, v in (data.get("variants") or {}).items()
        }
        self.file_path = file_path
        self.loaded_at = time.time()

    @property
    def version_key(self) -> str:
        """返回 name_vN 格式的版本标识"""
        return f"{self.name}_v{self.version}"

    def variant_names(self) -> list[str]:
        """已登记的变体名（不含基线）"""
        return list(self.variants.keys())

    def resolve_variant_content(self, variant: str | None) -> tuple[str, str]:
        """解析变体内容

        Returns:
            (实际使用的变体名("" = 基线), 内容文本)
            请求了未登记的变体时回退基线——坏请求头不应打断对话
        """
        if not variant:
            return "", self.content
        entry = self.variants.get(variant)
        if entry is None or not str(entry.get("content", "")).strip():
            logger.warning(f"Prompt '{self.name}' 无变体 '{variant}'，回退基线")
            return "", self.content
        return variant, str(entry["content"])

    def render(self, **kwargs) -> str:
        """渲染模板，填充变量

        Args:
            **kwargs: 模板变量

        Returns:
            渲染后的 Prompt 文本

        Raises:
            ValueError: 缺少必需变量
        """
        # 校验必需变量
        for var in self.variables:
            if var.get("required", False) and var["name"] not in kwargs:
                raise ValueError(f"Prompt '{self.name}' 缺少必需变量: {var['name']}")

        # 渲染
        rendered = self.content
        for key, value in kwargs.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))

        return rendered

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "variables": self.variables,
            "model_config": self.model_config,
            "blocks": self.blocks,
            "variants": self.variant_names(),
        }


class PromptManager:
    """Prompt 模板管理器

    从 prompts/ 目录加载所有 YAML 模板，
    支持按名称获取、热加载、版本追踪。
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._templates: dict[str, PromptTemplate] = {}
        self._blocks: dict[str, PromptBlock] = {}
        self._file_mtimes: dict[str, float] = {}
        self._load_all()

    def _load_all(self):
        """加载 prompts/*.yaml 模板与 prompts/blocks/*.yaml 可复用块"""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts 目录不存在: {self.prompts_dir}")
            return

        count = 0
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            self._load_file(yaml_file)
            count += 1

        for yml_file in self.prompts_dir.glob("*.yml"):
            self._load_file(yml_file)
            count += 1

        # P5: blocks 子目录（非递归 glob 不会自动覆盖到，需显式加载）
        block_count = 0
        blocks_dir = self.prompts_dir / "blocks"
        if blocks_dir.is_dir():
            for pattern in ("*.yaml", "*.yml"):
                for block_file in sorted(blocks_dir.glob(pattern)):
                    self._load_block_file(block_file)
                    block_count += 1

        logger.info(
            f"Prompt 模板加载完成: {count} 个模板文件 ({len(self._templates)} 模板), "
            f"{block_count} 个块文件 ({len(self._blocks)} 块)"
        )

    def _load_block_file(self, file_path: Path):
        """加载单个块 YAML（无效块告警跳过，不影响其他块）"""
        try:
            mtime = file_path.stat().st_mtime
            self._file_mtimes[str(file_path)] = mtime

            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning(f"跳过无效 Prompt 块文件: {file_path}")
                return

            block = PromptBlock(data, file_path=str(file_path))
            if not block.is_valid:
                logger.warning(
                    f"跳过无效 Prompt 块 '{block.name}' (kind={block.kind!r}): {file_path}"
                )
                return

            self._blocks[block.name] = block
            logger.debug(f"加载 Prompt 块: {block.name} [{block.kind}] <- {file_path.name}")

        except Exception as e:
            logger.error(f"加载 Prompt 块文件失败 {file_path}: {e}")

    def _load_file(self, file_path: Path):
        """加载单个 YAML 文件"""
        try:
            mtime = file_path.stat().st_mtime
            self._file_mtimes[str(file_path)] = mtime

            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "name" not in data:
                logger.warning(f"跳过无效 Prompt 文件: {file_path}")
                return

            template = PromptTemplate(data, file_path=str(file_path))
            self._templates[template.name] = template
            logger.debug(f"加载 Prompt: {template.name} v{template.version} <- {file_path.name}")

        except Exception as e:
            logger.error(f"加载 Prompt 文件失败 {file_path}: {e}")

    def get(self, name: str) -> PromptTemplate | None:
        """获取 Prompt 模板（带热加载检查）"""
        self._check_hot_reload(name)
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """获取并渲染 Prompt

        Args:
            name: Prompt 名称
            **kwargs: 模板变量

        Returns:
            渲染后的 Prompt 文本

        Raises:
            KeyError: Prompt 不存在
        """
        template = self.get(name)
        if not template:
            raise KeyError(f"Prompt '{name}' 不存在，可用: {list(self._templates.keys())}")
        return template.render(**kwargs)

    def get_version(self, name: str) -> str:
        """获取指定 Prompt 的版本号"""
        template = self.get(name)
        return template.version_key if template else ""

    def get_block(self, name: str) -> PromptBlock | None:
        """获取可复用块（带热加载检查）"""
        block = self._blocks.get(name)
        if block and block.file_path:
            self._check_hot_reload_path(block.file_path)
            block = self._blocks.get(name)  # 热加载后取新实例
        return block

    def list_blocks(self) -> list[dict]:
        """列出所有已加载的块"""
        return [
            {
                "name": b.name,
                "kind": b.kind,
                "intents": b.intents,
                "priority": b.priority,
                "description": b.description,
            }
            for b in self._blocks.values()
        ]

    def render_composed(
        self,
        name: str,
        *,
        variant: str | None = None,
        intent: str | None = None,
        **kwargs,
    ) -> str:
        """组合渲染：块 + 模板正文（或变体正文）+ 变量替换

        组装顺序：persona 块 → 模板正文 → rules 块 → few_shot 块。
        - 正文块来自模板声明的 blocks 列表，按声明顺序
        - few_shot 块由块库按 intent 标签自动挑选（intent=None 只保留通用块）
        - 变体只替换正文；块组合对基线与所有变体一视同仁

        Args:
            name: Prompt 名称
            variant: AB 变体名；None/"" 用基线，未登记的名回退基线
            intent: 意图标签（驱动 few-shot 选择）
            **kwargs: 模板变量

        Raises:
            KeyError: Prompt 不存在
        """
        template = self.get(name)
        if not template:
            raise KeyError(f"Prompt '{name}' 不存在，可用: {list(self._templates.keys())}")

        # 评审修复：引用块先过 mtime 热加载——否则块编辑对服务路径不可达
        # （get_block 才有检查，而生产组合只走这里）
        for block_name in template.blocks:
            block = self._blocks.get(block_name)
            if block and block.file_path:
                self._check_hot_reload_path(block.file_path)

        # 与 render() 对齐的必需变量校验（评审修复：此前组合渲染静默漏检）
        for var in template.variables:
            if var.get("required", False) and var["name"] not in kwargs:
                raise ValueError(f"Prompt '{name}' 缺少必需变量: {var['name']}")

        used_variant, body = template.resolve_variant_content(variant)
        sections: list[str] = []
        ordered_declared = sorted(
            (self._blocks[b] for b in template.blocks if b in self._blocks),
            key=lambda b: (b.priority, b.name),
        )
        missing = [b for b in template.blocks if b not in self._blocks]
        if missing:
            logger.warning(f"Prompt '{name}' 引用了不存在的块: {missing}")

        # 1) persona 在前
        sections.extend(b.content.strip() for b in ordered_declared if b.kind == "persona")
        # 2) 模板正文 / 变体正文
        sections.append(body.strip())
        # 3) rules 在正文后（约束紧贴任务内容）
        sections.extend(b.content.strip() for b in ordered_declared if b.kind == "rules")

        # few-shot：库按 intent 标签自动挑选
        few_shots = [
            block
            for block in sorted(self._blocks.values(), key=lambda b: (b.priority, b.name))
            if block.kind == "few_shot" and block.matches_intent(intent)
        ]
        sections.extend(block.content.strip() for block in few_shots)

        rendered = "\n\n".join(s for s in sections if s)
        if used_variant:
            logger.debug(f"Prompt '{name}' 使用变体 '{used_variant}'")

        # 与 render() 相同的变量语义（对组合后全文做一次替换）
        for key, value in kwargs.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    def effective_variant(self, name: str, requested: str | None) -> str:
        """解析实际生效的变体名（"" = 基线）——供调用方归因记录"""
        template = self.get(name)
        if not template:
            return ""
        used, _ = template.resolve_variant_content(requested)
        return used

    def _check_hot_reload(self, name: str):
        """检查模板文件是否变更，如果变更则重新加载"""
        template = self._templates.get(name)
        if not template or not template.file_path:
            return
        self._check_hot_reload_path(template.file_path)

    def _check_hot_reload_path(self, file_path: str):
        """按路径检查文件是否变更（模板与块共用）"""
        try:
            current_mtime = os.path.getmtime(file_path)
            cached_mtime = self._file_mtimes.get(file_path, 0)

            if current_mtime > cached_mtime:
                logger.info(f"检测到 Prompt 文件变更，重新加载: {file_path}")
                path = Path(file_path)
                if "/blocks/" in file_path.replace(os.sep, "/"):
                    self._load_block_file(path)
                else:
                    self._load_file(path)

        except OSError:
            pass

    def list_templates(self) -> list[dict]:
        """列出所有已加载的模板"""
        return [t.to_dict() for t in self._templates.values()]

    def reload_all(self):
        """强制重新加载所有模板与块"""
        self._templates.clear()
        self._blocks.clear()
        self._file_mtimes.clear()
        self._load_all()


# 全局单例
prompt_manager = PromptManager()
