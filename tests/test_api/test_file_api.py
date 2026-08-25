"""文件接口测试：/api/index_directory 的目录白名单

该端点曾接受任意服务器路径——本机任意可读 .txt/.md 都能被索引进向量库、
经对话检索外泄，故客户端传入的 directory_path 必须落在白名单根目录内。
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass
class _FakeIndexingResult:
    success: bool = True
    total_files: int = 0
    to_dict_fields: dict = field(default_factory=dict)

    def to_dict(self):
        return {"success": self.success, "total_files": self.total_files, **self.to_dict_fields}


@pytest.fixture
def index_sandbox(tmp_path, monkeypatch):
    """把工作目录切到沙箱，创建白名单根目录并 mock 掉真实索引服务"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "aiops-docs").mkdir()

    calls: list[str | None] = []

    class _FakeService:
        def index_directory(self, directory_path=None):
            calls.append(directory_path)
            return _FakeIndexingResult(total_files=1)

    import app.api.file as file_module

    monkeypatch.setattr(file_module, "vector_index_service", _FakeService())
    return calls


class TestIndexDirectoryAllowlist:
    def test_arbitrary_absolute_path_rejected(self, test_app, index_sandbox):
        resp = test_app.post("/api/index_directory", params={"directory_path": "/etc"})
        assert resp.status_code == 400
        assert "uploads" in resp.json()["detail"]

    def test_home_like_relative_escape_rejected(self, test_app, index_sandbox):
        resp = test_app.post("/api/index_directory", params={"directory_path": "../users-notes"})
        assert resp.status_code == 400

    def test_traversal_inside_name_rejected(self, test_app, index_sandbox):
        """resolve 后越出白名单根的前缀穿越同样拒绝"""
        (Path.cwd() / "uploads" / "nested").mkdir()
        resp = test_app.post(
            "/api/index_directory",
            params={"directory_path": "uploads/nested/../../outside"},
        )
        assert resp.status_code == 400

    def test_uploads_subpath_allowed(self, test_app, index_sandbox):
        (Path.cwd() / "uploads" / "sub").mkdir()
        resp = test_app.post("/api/index_directory", params={"directory_path": "uploads/sub"})
        assert resp.status_code == 200
        assert resp.json()["code"] == 200
        assert index_sandbox[-1] == "uploads/sub"

    def test_aiops_docs_root_allowed(self, test_app, index_sandbox):
        resp = test_app.post("/api/index_directory", params={"directory_path": "./aiops-docs"})
        assert resp.status_code == 200
        assert index_sandbox[-1] == "./aiops-docs"

    def test_default_none_skips_validation(self, test_app, index_sandbox):
        """不传路径时服务层走默认 uploads，无需校验"""
        resp = test_app.post("/api/index_directory")
        assert resp.status_code == 200
        assert index_sandbox[-1] is None
