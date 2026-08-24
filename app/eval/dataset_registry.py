"""金标评测集版本化注册器 —— 数据集也是被测物，必须可追溯

解决「金标集未版本化」：谁改了一行期望值、评测数字就失去可比性。

规范格式（envelope）：
    {"version": "v3", "sha256": "<cases 规范化哈希>", "updated_at": "...", "cases": [...]}

规则：
- load_versioned：拒载无版本文件（legacy 裸列表），哈希不符视为篡改直接报错
- read_cases：宽松读取，兼容 legacy 裸列表（供 evaluator 迁移期过渡）
- 版本号约定 "v<N>"；bump_version 负责递增，回填/修订金标必须换版本
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


class DatasetRegistryError(Exception):
    """数据集加载/校验失败"""


@dataclass(frozen=True)
class DatasetManifest:
    """一次成功加载数据集的元信息"""

    name: str
    path: str
    version: str
    sha256: str
    case_count: int


def canonical_hash(cases: list[dict]) -> str:
    """cases 的规范化哈希（键排序 + 紧凑序列化，跨机器稳定）"""
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stamp_dataset(cases: list[dict], version: str) -> dict[str, Any]:
    """构造带版本与内容哈希的数据集信封"""
    if not version:
        raise DatasetRegistryError("版本号不能为空")
    return {
        "version": version,
        "sha256": canonical_hash(cases),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": cases,
    }


def bump_version(version: str) -> str:
    """v2 → v3；非 vN 形态则从 v1 起"""
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return "v1"


def save_dataset(path: str | Path, cases: list[dict], version: str) -> DatasetManifest:
    """写入规范格式数据集文件"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = stamp_dataset(cases, version)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)
    logger.info(f"数据集已写入: {target} (version={version}, {len(cases)} 用例)")
    return _manifest_from(target.name, str(target), envelope)


def load_versioned(
    path: str | Path, expected_version: str | None = None
) -> tuple[list[dict], DatasetManifest]:
    """严格加载：必须是规范格式且内容哈希吻合（拒载无版本 / 被篡改文件）

    Returns:
        (用例列表, 元信息)
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetRegistryError(f"数据集不存在: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "version" not in raw or "cases" not in raw:
        raise DatasetRegistryError(
            f"{file_path.name} 缺少 version/cases 信封——金标集必须版本化后才能用于评测"
        )

    cases = raw["cases"]
    digest = raw.get("sha256", "")
    actual = canonical_hash(cases)
    if not digest:
        raise DatasetRegistryError(f"{file_path.name} 缺少内容哈希 sha256")
    if digest != actual:
        raise DatasetRegistryError(
            f"{file_path.name} 内容哈希不符（文件被改动但未重新登记？）: "
            f"记录={digest[:12]}… 实际={actual[:12]}…"
        )
    if expected_version is not None and raw["version"] != expected_version:
        raise DatasetRegistryError(
            f"{file_path.name} 版本不匹配: 期望 {expected_version}, 实际 {raw['version']}"
        )

    manifest = _manifest_from(file_path.name, str(file_path), raw)
    logger.debug(f"数据集加载: {manifest.name} {manifest.version} ({manifest.case_count} 用例)")
    return cases, manifest


def read_cases(path: str | Path) -> list[dict]:
    """宽松读取：规范信封或 legacy 裸列表都接受（消费方迁移期使用）

    注意：legacy 文件无版本无哈希，只可用于展示/统计，不可作为评测门禁依据。
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "cases" in raw:
        return list(raw["cases"])
    if isinstance(raw, list):
        logger.warning(f"{file_path.name} 为未版本化 legacy 格式，建议运行 register 迁移")
        return list(raw)
    return []


def _manifest_from(name: str, path: str, envelope: dict[str, Any]) -> DatasetManifest:
    return DatasetManifest(
        name=name,
        path=path,
        version=str(envelope.get("version", "")),
        sha256=str(envelope.get("sha256", "")),
        case_count=len(envelope.get("cases", [])),
    )
