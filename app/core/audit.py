"""请求审计日志

将每次请求的关键信息写入 JSON Lines 文件，
支持后续分析和合规审查。
"""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


class AuditLogger:
    """审计日志记录器

    写入 JSON Lines 格式（每行一条记录）到 logs/audit.jsonl。
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_dir / "audit.jsonl"
        self.cost_file = self.log_dir / "cost.jsonl"
        logger.info(f"审计日志初始化: {self.audit_file}")

    def log_request(
        self,
        request_id: str,
        session_id: str = "default",
        intent: str = "",
        scene: str = "",
        model_used: str = "",
        prompt_version: str = "",
        degradation_level: str = "none",
        total_tokens: int = 0,
        total_cost: float = 0,
        latency_ms: float = 0,
        error: str | None = None,
        extra: dict | None = None,
    ):
        """记录请求审计日志"""
        record = {
            "timestamp": time.time(),
            "type": "request",
            "request_id": request_id,
            "session_id": session_id,
            "intent": intent,
            "scene": scene,
            "model_used": model_used,
            "prompt_version": prompt_version,
            "degradation_level": degradation_level,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }
        if extra:
            record.update(extra)
        self._write(self.audit_file, record)

    def log_tool_call(
        self,
        request_id: str,
        tool_name: str,
        params: dict | None = None,
        result_status: str = "success",
        latency_ms: float = 0,
        error: str | None = None,
    ):
        """记录工具调用审计日志"""
        record = {
            "timestamp": time.time(),
            "type": "tool_call",
            "request_id": request_id,
            "tool_name": tool_name,
            "params_summary": self._summarize_params(params),
            "result_status": result_status,
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }
        self._write(self.audit_file, record)

    def log_cost(self, cost_record: dict):
        """记录成本到独立文件"""
        cost_record["timestamp"] = time.time()
        self._write(self.cost_file, cost_record)

    def log_trace(self, trace: Any):
        """记录完整的 Trace 摘要"""
        record = {
            "timestamp": time.time(),
            "type": "trace_summary",
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "total_duration_ms": round(trace.total_duration_ms, 2),
            "total_tokens": trace.total_tokens,
            "total_cost": round(trace.total_cost, 6),
            "span_count": len(trace.spans),
            "generation_count": len(trace.generations),
        }
        self._write(self.audit_file, record)

    def _write(self, filepath: Path, record: dict):
        """追加写入一条 JSON Lines 记录"""
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"审计日志写入失败: {e}")

    def _summarize_params(self, params: dict | None) -> str:
        """参数摘要（避免记录过长内容）"""
        if not params:
            return ""
        summary = {}
        for k, v in params.items():
            sv = str(v)
            summary[k] = sv[:100] + "..." if len(sv) > 100 else sv
        return json.dumps(summary, ensure_ascii=False)


# 全局单例
audit_logger = AuditLogger()
