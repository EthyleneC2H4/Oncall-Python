"""Langfuse 全链路追踪

每次请求生成一个 Trace，记录完整调用链路：
路由 → 检索 → LLM → 工具 → 回答

支持与 Prompt 版本关联（阶段 2）和成本追踪（阶段 5）。
"""

import time
import uuid
from contextlib import contextmanager
from typing import Any

from loguru import logger


class TraceSpan:
    """轻量级 Span，记录单个操作的耗时和元数据"""

    def __init__(self, name: str, trace_id: str, metadata: dict | None = None):
        self.span_id = str(uuid.uuid4())[:8]
        self.name = name
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "running"
        self.output: Any = None

    def end(self, status: str = "success", output: Any = None):
        self.end_time = time.time()
        self.status = status
        self.output = output

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "output": str(self.output)[:500] if self.output else None,
        }


class Trace:
    """请求级 Trace，包含多个 Span"""

    def __init__(self, request_id: str | None = None, session_id: str = "default"):
        self.trace_id = request_id or str(uuid.uuid4())
        self.session_id = session_id
        self.start_time = time.time()
        self.spans: list[TraceSpan] = []
        self.generations: list[dict] = []
        self.metadata: dict = {}

    def span(self, name: str, metadata: dict | None = None) -> TraceSpan:
        """创建并返回一个新 Span"""
        s = TraceSpan(name=name, trace_id=self.trace_id, metadata=metadata)
        self.spans.append(s)
        return s

    @contextmanager
    def span_context(self, name: str, metadata: dict | None = None):
        """上下文管理器形式的 Span"""
        s = self.span(name, metadata)
        try:
            yield s
            s.end("success")
        except Exception as e:
            s.end("error", output=str(e))
            raise

    def record_generation(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
        cost: float = 0,
        prompt_name: str = "",
        prompt_version: str = "",
    ):
        """记录一次 LLM Generation"""
        gen = {
            "trace_id": self.trace_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost": round(cost, 6),
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "timestamp": time.time(),
        }
        self.generations.append(gen)
        logger.debug(
            f"[Trace {self.trace_id[:8]}] Generation: model={model}, "
            f"tokens={input_tokens}+{output_tokens}, cost={cost:.4f}"
        )

    @property
    def total_duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(g.get("input_tokens", 0) + g.get("output_tokens", 0) for g in self.generations)

    @property
    def total_cost(self) -> float:
        return float(sum(g.get("cost", 0) for g in self.generations))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "spans": [s.to_dict() for s in self.spans],
            "generations": self.generations,
            "metadata": self.metadata,
        }

    def summary_log(self):
        """输出结构化 Trace 摘要日志"""
        logger.info(
            f"[Trace {self.trace_id[:8]}] "
            f"session={self.session_id} "
            f"duration={self.total_duration_ms:.0f}ms "
            f"spans={len(self.spans)} "
            f"generations={len(self.generations)} "
            f"tokens={self.total_tokens} "
            f"cost={self.total_cost:.4f}"
        )


class TracingManager:
    """全局 Tracing 管理器

    管理当前请求的 Trace 上下文。
    通过 middleware 在请求入口创建 Trace，在请求结束时记录。
    """

    def __init__(self):
        self._current_traces: dict[str, Trace] = {}
        self.enabled = True

    def create_trace(self, request_id: str | None = None, session_id: str = "default") -> Trace:
        """创建新的 Trace"""
        trace = Trace(request_id=request_id, session_id=session_id)
        self._current_traces[trace.trace_id] = trace
        logger.debug(f"[Tracing] Created trace {trace.trace_id[:8]}")
        return trace

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._current_traces.get(trace_id)

    def finish_trace(self, trace_id: str):
        """完成并记录 Trace"""
        trace = self._current_traces.pop(trace_id, None)
        if trace:
            trace.summary_log()
            # 将 Trace 写入审计日志
            from app.core.audit import audit_logger

            audit_logger.log_trace(trace)

    @property
    def active_traces(self) -> int:
        return len(self._current_traces)


# 全局单例
tracing_manager = TracingManager()
