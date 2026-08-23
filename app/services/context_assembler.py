"""上下文工程 - 动态组装每轮调用的最优上下文

实现 Token 优先级策略：
- 高优先级（永不丢失）: 系统提示词 + 当前查询
- 中优先级（可精简）: KG 分析结果 + RAG 检索段落
- 低优先级（可压缩）: 早期对话历史 → 摘要压缩
"""

from loguru import logger


class ContextAssembler:
    """动态组装每轮调用的最优上下文"""

    # 各部分的 Token 预算（近似，1 个中文字符约 1.5 token）
    MAX_KG_CHARS = 1200
    MAX_RAG_CHARS = 2000
    HISTORY_COMPRESS_THRESHOLD = 6  # 超过此轮数时压缩早期历史

    def assemble(
        self,
        query: str,
        history: list | None = None,
        kg_context: str = "",
        rag_context: str = "",
    ) -> dict:
        """
        组装最优上下文

        Args:
            query: 当前用户查询
            history: 对话历史消息列表
            kg_context: 知识图谱分析结果
            rag_context: RAG 检索结果

        Returns:
            dict: 组装后的上下文，包含各优先级部分
        """
        context = {
            "query": query,
            "kg_analysis": "",
            "rag_context": "",
            "history_summary": "",
            "recent_history": [],
        }

        # 中优先级：KG 分析（截断保护）
        if kg_context:
            context["kg_analysis"] = self._truncate(kg_context, self.MAX_KG_CHARS)

        # 中优先级：RAG 检索（截断保护）
        if rag_context:
            context["rag_context"] = self._truncate(rag_context, self.MAX_RAG_CHARS)

        # 低优先级：历史对话压缩
        if history and len(history) > self.HISTORY_COMPRESS_THRESHOLD:
            # 早期历史压缩为摘要
            early = history[:-4]
            recent = history[-4:]
            context["history_summary"] = self._summarize_history(early)
            context["recent_history"] = recent
            logger.debug(
                f"上下文组装: 历史 {len(history)} 条 → 压缩早期 {len(early)} 条 + 保留最近 {len(recent)} 条"
            )
        elif history:
            context["recent_history"] = history

        return context

    def format_experience_context(
        self,
        kg_context: str = "",
        rag_context: str = "",
    ) -> str:
        """格式化经验上下文（用于 Planner）

        Args:
            kg_context: 知识图谱分析结果
            rag_context: RAG 检索的文档

        Returns:
            str: 格式化的上下文文本
        """
        parts = []

        if kg_context:
            parts.append(
                "## 知识图谱分析（结构化关联信息）\n\n"
                f"{self._truncate(kg_context, self.MAX_KG_CHARS)}\n\n---"
            )

        if rag_context:
            parts.append(
                "## 相关经验文档\n\n" f"{self._truncate(rag_context, self.MAX_RAG_CHARS)}\n\n---"
            )

        return "\n\n".join(parts)

    def _truncate(self, text: str, max_chars: int) -> str:
        """截断文本到指定字符数"""
        if len(text) <= max_chars:
            return text
        logger.debug(f"上下文截断: {len(text)} → {max_chars} 字符")
        return text[:max_chars] + "\n...(内容已截断)"

    def _summarize_history(self, messages: list) -> str:
        """将早期对话历史压缩为摘要（本地规则，不调用 LLM）

        提取每轮对话的核心信息，压缩为简短摘要。
        """
        if not messages:
            return ""

        summaries = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            # 提取每条消息的前 80 个字符
            preview = content[:80].replace("\n", " ")
            if len(content) > 80:
                preview += "..."
            summaries.append(f"[{role}] {preview}")

        return "[历史摘要]\n" + "\n".join(summaries)


# 全局单例
context_assembler = ContextAssembler()
