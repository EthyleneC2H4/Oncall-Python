"""金标集版本化注册器测试：信封读写 / 拒载无版本 / 篡改检测"""

import json

import pytest

from app.eval.dataset_registry import (
    DatasetRegistryError,
    bump_version,
    canonical_hash,
    load_versioned,
    read_cases,
    save_dataset,
    stamp_dataset,
)

CASES = [
    {"id": "TC001", "query": "CPU 90%", "required_evidence": ["内存泄漏"]},
    {"id": "TC002", "query": "磁盘写满", "required_evidence": ["清理日志"]},
]


class TestStampAndSave:
    def test_stamp_contains_version_and_hash(self):
        envelope = stamp_dataset(CASES, "v1")
        assert envelope["version"] == "v1"
        assert envelope["sha256"] == canonical_hash(CASES)
        assert envelope["cases"] == CASES

    def test_empty_version_rejected(self):
        with pytest.raises(DatasetRegistryError):
            stamp_dataset(CASES, "")

    def test_save_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "ds.json"
        save_dataset(path, CASES, "v2")

        cases, manifest = load_versioned(path)
        assert cases == CASES
        assert manifest.version == "v2"
        assert manifest.case_count == 2
        assert manifest.name == "ds.json"

    def test_hash_changes_when_cases_change(self):
        mutated = [dict(c) for c in CASES]
        mutated[0]["required_evidence"] = ["别的根因"]
        assert canonical_hash(mutated) != canonical_hash(CASES)


class TestLoadStrictness:
    def test_unversioned_bare_list_rejected(self, tmp_path):
        """金标集必须版本化：legacy 裸列表拒载（核心契约）"""
        path = tmp_path / "bare.json"
        path.write_text(json.dumps(CASES, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(DatasetRegistryError, match="缺少.*version|版本化"):
            load_versioned(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetRegistryError, match="不存在"):
            load_versioned(tmp_path / "ghost.json")

    def test_tampered_content_rejected(self, tmp_path):
        """文件被手改但未重新登记 → 哈希不符直接报错"""
        path = tmp_path / "tamper.json"
        save_dataset(path, CASES, "v1")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["cases"][0]["query"] = "被偷偷改过的查询"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(DatasetRegistryError, match="哈希不符"):
            load_versioned(path)

    def test_expected_version_mismatch(self, tmp_path):
        path = tmp_path / "ver.json"
        save_dataset(path, CASES, "v3")
        with pytest.raises(DatasetRegistryError, match="版本不匹配"):
            load_versioned(path, expected_version="v2")


class TestReadCasesLenient:
    def test_envelope_and_bare_list_both_accepted(self, tmp_path):
        envelope_path = tmp_path / "envelope.json"
        bare_path = tmp_path / "bare.json"
        save_dataset(envelope_path, CASES, "v1")
        bare_path.write_text(json.dumps(CASES, ensure_ascii=False), encoding="utf-8")

        assert read_cases(envelope_path) == CASES
        assert read_cases(bare_path) == CASES

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_cases(tmp_path / "nope.json") == []


def test_bump_version():
    assert bump_version("v1") == "v2"
    assert bump_version("v9") == "v10"
    assert bump_version("") == "v1"  # legacy 无版本 → 从 v1 起
    assert bump_version("weird") == "v1"
