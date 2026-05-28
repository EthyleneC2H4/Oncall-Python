"""Prompt 版本化管理

功能：
- 从 YAML 文件加载 Prompt 模板
- 支持版本管理和变量校验
- 文件变更自动热加载（watchdog）
- 与 Trace 关联 prompt_name + prompt_version
"""

import os
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class PromptTemplate:
    """单个 Prompt 模板"""

    def __init__(self, data: dict, file_path: str = ""):
        self.name: str = data.get("name", "")
        self.version: str = str(data.get("version", "1"))
        self.description: str = data.get("description", "")
        self.variables: list[dict] = data.get("variables", [])
        self.model_config: dict = data.get("model_config", {})
        self.content: str = data.get("content", "")
        self.file_path = file_path
        self.loaded_at = time.time()

    @property
    def version_key(self) -> str:
        """返回 name_vN 格式的版本标识"""
        return f"{self.name}_v{self.version}"

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
                raise ValueError(
                    f"Prompt '{self.name}' 缺少必需变量: {var['name']}"
                )

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
        }


class PromptManager:
    """Prompt 模板管理器

    从 prompts/ 目录加载所有 YAML 模板，
    支持按名称获取、热加载、版本追踪。
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._templates: dict[str, PromptTemplate] = {}
        self._file_mtimes: dict[str, float] = {}
        self._load_all()

    def _load_all(self):
        """加载 prompts/ 目录下的所有 YAML 文件"""
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

        logger.info(f"Prompt 模板加载完成: {count} 个文件, {len(self._templates)} 个模板")

    def _load_file(self, file_path: Path):
        """加载单个 YAML 文件"""
        try:
            mtime = file_path.stat().st_mtime
            self._file_mtimes[str(file_path)] = mtime

            with open(file_path, "r", encoding="utf-8") as f:
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

    def _check_hot_reload(self, name: str):
        """检查文件是否变更，如果变更则重新加载"""
        template = self._templates.get(name)
        if not template or not template.file_path:
            return

        try:
            current_mtime = os.path.getmtime(template.file_path)
            cached_mtime = self._file_mtimes.get(template.file_path, 0)

            if current_mtime > cached_mtime:
                logger.info(f"检测到 Prompt 文件变更，重新加载: {template.file_path}")
                self._load_file(Path(template.file_path))

        except OSError:
            pass

    def list_templates(self) -> list[dict]:
        """列出所有已加载的模板"""
        return [t.to_dict() for t in self._templates.values()]

    def reload_all(self):
        """强制重新加载所有模板"""
        self._templates.clear()
        self._file_mtimes.clear()
        self._load_all()


# 全局单例
prompt_manager = PromptManager()
