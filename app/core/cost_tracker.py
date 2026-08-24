"""成本追踪

记录每次 LLM 调用的 Token 用量和成本，
支持按 session / 场景 / 模型 聚合。
"""

from collections import defaultdict
from typing import Any

from loguru import logger

from app.core.audit import audit_logger

# OpenRouter 定价（每千 Token，单位: 美元）
# 数据来源: GET https://openrouter.ai/api/v1/models （2026-08 查询）
MODEL_PRICING: dict[str, dict[str, float]] = {
    # 主力模型：NVIDIA Nemotron 3.5 Lightning ($0.08/$0.20 每 1M token)
    "nvidia/nemotron-3.5-lightning": {"input": 0.00008, "output": 0.0002},
    # 弱模型层（路由/改写等轻任务）
    "nvidia/nemotron-3-nano-30b-a3b": {"input": 0.00005, "output": 0.0002},
    # 免费档变体
    "nvidia/nemotron-3.5-lightning:free": {"input": 0, "output": 0},
    "nvidia/nemotron-3-nano-30b-a3b:free": {"input": 0, "output": 0},
    # 本地推理组件：不产生 API 费用
    "local/bge-large-zh-v1.5": {"input": 0, "output": 0},
    "local/bge-reranker-base": {"input": 0, "output": 0},
}

# 未登记模型的兜底定价（按主力模型计，宁可高估不高估）
DEFAULT_PRICING: dict[str, float] = MODEL_PRICING["nvidia/nemotron-3.5-lightning"]


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
        # P5 Prompt AB 归因：variant 名（"" = 基线）→ 运行数与去重会话
        self._prompt_variants: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"runs": 0, "sessions": set()}
        )

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
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

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
            f"cost=${total_cost:.6f} scene={scene}"
        )

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
        }

    def mark_prompt_variant(self, variant: str, session_id: str = "", request_id: str = "") -> None:
        """记录一次 Prompt 变体运行（P5 AB 归因）

        不记 token/费用——那是 record 的职责；这里只做「哪个变体、跑了
        多少次、覆盖多少会话」的分组计数，供 Phase 4 judge 按
        base vs variant 分组做 pairwise 比较。

        Args:
            variant: 变体名；空串表示基线（同样计数，作为对照组）
            session_id: 会话 ID（去重统计）
            request_id: 请求 ID（预留，暂不聚合）
        """
        key = variant or ""
        stats = self._prompt_variants[key]
        stats["runs"] += 1
        if session_id:
            stats["sessions"].add(session_id)

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
            "by_scene": {scene: round(cost, 4) for scene, cost in self._by_scene.items()},
            "prompt_variants": {
                (variant or "base"): {
                    "runs": stats["runs"],
                    "sessions": len(stats["sessions"]),
                }
                for variant, stats in self._prompt_variants.items()
            },
        }

    def reset(self):
        """重置统计"""
        self._total_cost = 0.0
        self._total_tokens = 0
        self._by_model.clear()
        self._by_scene.clear()
        self._prompt_variants.clear()


# 全局单例
cost_tracker = CostTracker()
