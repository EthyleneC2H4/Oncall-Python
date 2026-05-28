"""成本追踪

记录每次 LLM 调用的 Token 用量和成本，
支持按 session / 场景 / 模型 聚合。
"""

import time
from collections import defaultdict
from typing import Any

from loguru import logger

from app.core.audit import audit_logger


# DashScope 定价（每千 Token，单位: 元）
MODEL_PRICING = {
    "qwen-max": {"input": 0.02, "output": 0.06},
    "qwen-plus": {"input": 0.004, "output": 0.012},
    "qwen-turbo": {"input": 0.002, "output": 0.006},
    "text-embedding-v4": {"input": 0.0007, "output": 0},
    "gte-rerank": {"input": 0.001, "output": 0},
}


class CostTracker:
    """成本追踪器"""

    def __init__(self):
        # 内存中的成本统计
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._by_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        )
        self._by_scene: dict[str, float] = defaultdict(float)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        scene: str = "unknown",
        request_id: str = "",
        session_id: str = "",
    ) -> dict:
        """记录一次 LLM 调用的成本

        Args:
            model: 模型名称
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            scene: 业务场景 (chat/aiops/multi_diagnose/eval)
            request_id: 请求 ID
            session_id: 会话 ID

        Returns:
            {"input_cost": float, "output_cost": float, "total_cost": float}
        """
        pricing = MODEL_PRICING.get(model, {"input": 0.02, "output": 0.06})

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        # 更新内存统计
        self._total_cost += total_cost
        self._total_tokens += input_tokens + output_tokens
        self._by_model[model]["input_tokens"] += input_tokens
        self._by_model[model]["output_tokens"] += output_tokens
        self._by_model[model]["cost"] += total_cost
        self._by_model[model]["calls"] += 1
        self._by_scene[scene] += total_cost

        # 写入审计日志
        cost_record = {
            "request_id": request_id,
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "scene": scene,
        }
        audit_logger.log_cost(cost_record)

        logger.debug(
            f"[Cost] model={model} tokens={input_tokens}+{output_tokens} "
            f"cost={total_cost:.4f}元 scene={scene}"
        )

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
        }

    def get_summary(self) -> dict:
        """获取成本汇总"""
        return {
            "total_cost": round(self._total_cost, 4),
            "total_tokens": self._total_tokens,
            "by_model": {
                model: {
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "cost": round(stats["cost"], 4),
                    "calls": stats["calls"],
                }
                for model, stats in self._by_model.items()
            },
            "by_scene": {
                scene: round(cost, 4)
                for scene, cost in self._by_scene.items()
            },
        }

    def reset(self):
        """重置统计"""
        self._total_cost = 0.0
        self._total_tokens = 0
        self._by_model.clear()
        self._by_scene.clear()


# 全局单例
cost_tracker = CostTracker()
