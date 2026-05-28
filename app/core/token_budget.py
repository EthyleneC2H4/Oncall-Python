"""Token 预算管理

调用前估算 Token 数量，超预算时按优先级降级：
1. 删除低相关 RAG 片段（rerank score 最低的）
2. 压缩早期历史消息（保留最近 3 轮）
3. 减少工具 Schema（只保留当前任务相关的）
4. 降低 max_tokens
5. 拒绝执行并提示用户缩小范围
"""

from loguru import logger


class TokenBudgetManager:
    """Token 预算管理器"""

    # 各模型的 Token 上限
    MODEL_LIMITS = {
        "qwen-max": 30000,      # 预留 2000 给输出
        "qwen-plus": 30000,
        "qwen-turbo": 6000,
    }

    # 默认预算分配
    BUDGET_ALLOCATION = {
        "system_prompt": 0.10,    # 10% 给系统提示词
        "context": 0.50,          # 50% 给上下文（KG + RAG + 历史）
        "tools": 0.15,            # 15% 给工具描述
        "output": 0.25,           # 25% 给输出
    }

    # 中文近似：1 个字符 ≈ 1.5 token
    CHARS_PER_TOKEN = 0.67  # 1 token ≈ 0.67 个中文字符

    def __init__(self):
        pass

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 Token 数量（近似值）"""
        if not text:
            return 0
        # 中文字符 ≈ 1.5 token/char，英文 ≈ 0.25 token/word
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.4)

    def get_budget(self, model: str = "qwen-max") -> dict:
        """获取模型的 Token 预算分配"""
        total = self.MODEL_LIMITS.get(model, 30000)
        return {
            part: int(total * ratio)
            for part, ratio in self.BUDGET_ALLOCATION.items()
        }

    def trim_context(
        self,
        rag_context: str,
        kg_context: str = "",
        history: list | None = None,
        model: str = "qwen-max",
    ) -> dict:
        """根据预算裁剪上下文

        降级顺序：
        1. 截断低优先级 RAG 片段
        2. 压缩历史消息
        3. 截断 KG 上下文

        Returns:
            {"rag_context": str, "kg_context": str, "history": list, "trimmed": bool, "level": str}
        """
        budget = self.get_budget(model)
        context_budget = budget["context"]

        # 估算当前 Token
        rag_tokens = self.estimate_tokens(rag_context)
        kg_tokens = self.estimate_tokens(kg_context)
        history_tokens = sum(
            self.estimate_tokens(str(m)) for m in (history or [])
        )
        total_tokens = rag_tokens + kg_tokens + history_tokens

        result = {
            "rag_context": rag_context,
            "kg_context": kg_context,
            "history": history or [],
            "trimmed": False,
            "level": "none",
            "original_tokens": total_tokens,
            "budget_tokens": context_budget,
        }

        if total_tokens <= context_budget:
            return result

        logger.info(
            f"Token 超预算: {total_tokens} > {context_budget}, 启动降级"
        )
        result["trimmed"] = True

        # Level 1: 截断 RAG 上下文
        if rag_tokens > context_budget * 0.5:
            max_rag_chars = int(context_budget * 0.5 * self.CHARS_PER_TOKEN)
            result["rag_context"] = rag_context[:max_rag_chars] + "\n...(RAG 内容已截断以适应 Token 预算)"
            result["level"] = "rag_truncated"
            rag_tokens = self.estimate_tokens(result["rag_context"])
            logger.info(f"Level 1: RAG 截断至 {max_rag_chars} 字符")

        # Level 2: 压缩历史
        if history and len(history) > 4:
            result["history"] = history[-4:]
            result["level"] = "history_compressed"
            history_tokens = sum(
                self.estimate_tokens(str(m)) for m in result["history"]
            )
            logger.info(f"Level 2: 历史压缩至最近 4 条")

        # Level 3: 截断 KG 上下文
        total_tokens = rag_tokens + kg_tokens + history_tokens
        if total_tokens > context_budget and kg_tokens > 0:
            max_kg_chars = int(context_budget * 0.15 * self.CHARS_PER_TOKEN)
            result["kg_context"] = kg_context[:max_kg_chars] + "\n...(KG 内容已截断)"
            result["level"] = "kg_truncated"
            logger.info(f"Level 3: KG 截断至 {max_kg_chars} 字符")

        result["final_tokens"] = self.estimate_tokens(
            result["rag_context"] + result["kg_context"]
        ) + sum(self.estimate_tokens(str(m)) for m in result["history"])

        return result

    def check_budget(self, text: str, model: str = "qwen-max") -> dict:
        """检查文本是否在预算内

        Returns:
            {"within_budget": bool, "estimated_tokens": int, "budget": int, "utilization": float}
        """
        budget = self.get_budget(model)
        total_budget = sum(budget.values())
        estimated = self.estimate_tokens(text)
        return {
            "within_budget": estimated <= total_budget,
            "estimated_tokens": estimated,
            "budget": total_budget,
            "utilization": round(estimated / total_budget, 4) if total_budget > 0 else 0,
        }


# 全局单例
token_budget_manager = TokenBudgetManager()
