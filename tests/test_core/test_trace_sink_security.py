"""工具痕迹 sink 安全测试：敏感键脱敏 / 文件权限收紧

评审修复回归：tools.jsonl 是明文落盘的完整实参记录，
凭据不得原样入盘，文件不得对同机其他用户可读。
"""

import json
import os
import stat

from app.core.trace_sink import ToolTraceSink, _redact


class TestRedaction:
    def test_sensitive_keys_masked(self):
        red = _redact({"password": "hunter2", "api_key": "sk-x", "query": "cpu 高"})
        assert red["password"] == "***"
        assert red["api_key"] == "***"
        assert red["query"] == "cpu 高"  # 普通键不受影响

    def test_recursive_into_nested_containers(self):
        red = _redact({"opts": {"auth_token": "t", "keep": ["a", {"secret_value": 1}]}})
        assert red["opts"]["auth_token"] == "***"
        assert red["opts"]["keep"][0] == "a"
        assert red["opts"]["keep"][1]["secret_value"] == "***"

    def test_case_insensitive_key_match(self):
        assert _redact({"Authorization": "Bearer x"})["Authorization"] == "***"


class TestFilePermissions:
    def test_written_trace_owner_only_and_redacted(self, tmp_path):
        sink = ToolTraceSink(traces_dir=str(tmp_path / "traces"))
        sink.record("retrieve_knowledge", {"query": "cpu", "password": "p"})

        trace = tmp_path / "traces" / "tools.jsonl"
        mode = stat.S_IMODE(os.stat(trace).st_mode)
        assert mode & 0o077 == 0  # 组/其他无任何权限

        (entry,) = [
            json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
        ]
        assert entry["args"]["password"] == "***"
        assert entry["args"]["query"] == "cpu"

    def test_dir_permissions_tightened(self, tmp_path):
        d = tmp_path / "traces"
        d.mkdir(mode=0o755)  # 预先宽松创建，sink 应收紧
        ToolTraceSink(traces_dir=str(d)).record("t", {})
        assert stat.S_IMODE(os.stat(d).st_mode) & 0o077 == 0
