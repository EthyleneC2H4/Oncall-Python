"""工具调用痕迹 sink —— BFCL 式评测的数据源

audit.jsonl 出于安全只记 params_summary（截断摘要），不可用于
参数级评测；本 sink 把完整实参另落 data/traces/tools.jsonl，
供 app/eval/tool_eval.py 离线回放评分。

- 每行一条：{timestamp, request_id, session_id, tool_name, args, ok}
- 敏感键名脱敏后落盘（password/token/secret 等 → ***），文件权限 0600
- config.tool_trace_enabled 开关（默认开，写失败静默——观测不拖垮主流程）
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from loguru import logger

# 按键名脱敏的敏感片段（大小写不敏感；递归作用于嵌套 dict/list）
_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


def _redact(value: Any) -> Any:
    """按敏感键名递归脱敏——痕迹文件是明文落盘，凭据绝不原样入盘"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            k_lower = str(k).lower()
            out[str(k)] = "***" if any(p in k_lower for p in _SENSITIVE_KEY_PARTS) else _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class ToolTraceSink:
    """JSONL 追加式工具调用记录器（进程内单例）"""

    def __init__(self, traces_dir: str | None = None):
        # 显式传入目录的实例（测试用）固定路径；默认实例每次写入时读 config，
        # 保证配置变更无需重建单例
        self._fixed_dir = traces_dir
        self._lock = threading.Lock()

    @property
    def trace_file(self) -> str:
        return os.path.join(self._resolve_dir(), "tools.jsonl")

    def _resolve_dir(self) -> str:
        if self._fixed_dir is not None:
            return self._fixed_dir
        from app.config import config

        return config.traces_dir

    def record(
        self,
        tool_name: str,
        args: dict[str, Any] | None,
        *,
        request_id: str = "",
        session_id: str = "",
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        """追加一条工具调用记录（绝不抛异常）

        显式指定目录的实例（测试用）不受全局开关约束；
        默认单例经 config.tool_trace_enabled 门控。
        """
        try:
            gated = self._fixed_dir is None
            if gated:
                from app.config import config

                if not getattr(config, "tool_trace_enabled", True):
                    return
            entry = {
                "timestamp": time.time(),
                "request_id": request_id,
                "session_id": session_id,
                "tool_name": tool_name,
                "args": _redact(args or {}),
                "ok": ok,
            }
            if error:
                entry["error"] = error[:300]
            directory = self._resolve_dir()
            with self._lock:
                os.makedirs(directory, exist_ok=True)
                try:
                    os.chmod(directory, 0o700)
                except OSError:  # noqa: PERF203 - 权限收紧失败不阻断观测
                    pass
                path = os.path.join(directory, "tools.jsonl")
                created = not os.path.exists(path)
                with open(path, "a", encoding="utf-8") as f:
                    if created:
                        try:
                            os.chmod(path, 0o600)
                        except OSError:  # noqa: PERF203
                            pass
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 - 观测旁路失败不影响业务
            logger.debug(f"工具痕迹写入失败（忽略）: {e}")


# 全局单例（目录随 config.traces_dir 动态解析）
tool_trace_sink = ToolTraceSink()
