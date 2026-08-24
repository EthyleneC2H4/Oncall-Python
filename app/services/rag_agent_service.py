"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 LangGraph + OpenRouter（OpenAI 兼容模式）接入 LLM，
支持真正的流式输出和更好的模型适配。

Agent 构造、双通道流式与工具调用事件已迁移至
app.agent.runtime.react_runtime.ReActRuntime；本模块为薄门面，
保留旧的 query / query_stream / 会话管理接口（/api/chat 契约不变）。
"""

from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger

from app.agent.runtime import EventType, ReActRuntime
from app.config import config


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + OpenRouter 接入"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.runtime = ReActRuntime(
            model_name=self.model_name,
            temperature=0.7,
            streaming=streaming,
            system_prompt=self.system_prompt,
        )

        logger.info(
            f"RAG Agent 服务初始化完成 (OpenRouter), model={self.model_name}, streaming={streaming}"
        )

    def _build_system_prompt(self, variant: str | None = None) -> str:
        """
        构建结构化系统提示词（P5：块组合 + AB 变体）

        优先从 Prompt 版本化管理器组合渲染（persona/rules 块 + 正文），
        fallback 到硬编码。

        Args:
            variant: AB 变体名（X-Prompt-Variant 请求头）；未登记名回退基线

        Returns:
            str: 系统提示词
        """
        try:
            from app.core.prompt_manager import prompt_manager

            template = prompt_manager.get("system_prompt")
            if template:
                composed = prompt_manager.render_composed("system_prompt", variant=variant)
                effective = prompt_manager.effective_variant("system_prompt", variant)
                logger.debug(
                    f"加载 Prompt 模板: system_prompt v{template.version}"
                    + (f" (variant={effective})" if effective else "")
                )
                return composed
        except Exception as e:
            logger.debug(f"Prompt 模板加载失败，使用内置: {e}")

        from textwrap import dedent

        return dedent("""
            ## 角色
            你是一个智能运维助手（AIOps Agent），专注于基于知识图谱和文档检索的故障诊断与告警分析。

            ## 目标
            - 快速定位告警根因
            - 提供可执行的处置方案
            - 预测潜在的告警级联风险

            ## 约束
            - 优先使用知识图谱工具（query_alert_graph）查询告警关联关系
            - 使用文档检索工具（retrieve_knowledge）获取详细操作步骤作为补充
            - 如果检索结果不相关，明确告知用户"未找到相关知识"
            - 不要编造不存在的告警类型或处置方案
            - 基于事实回答，不确定时明确说明

            ## 执行流
            1. 分析用户输入，提取告警关键词
            2. 调用 query_alert_graph 查询告警的根因、处置和级联关系
            3. 调用 retrieve_knowledge 获取详细的操作文档
            4. 综合图谱分析和文档内容给出结构化回答

            ## 输出格式
            回答应包含：问题诊断 → 根因分析 → 处置建议 → 风险提示
            保持友好、专业的语气，回答简洁明了，重点突出。
        """).strip()

    def _resolve_prompt_variant(self, requested: str | None) -> str:
        """解析实际生效的 Prompt 变体（"" = 基线）；解析失败按基线处理

        AB 归因与 SSE complete 字段均由 runtime 层按实际使用的图下发
        （评审修复：此处不再预标记，未生效的变体不会被计入分组）
        """
        if not requested:
            return ""
        try:
            from app.core.prompt_manager import prompt_manager

            return prompt_manager.effective_variant("system_prompt", requested)
        except Exception as e:  # noqa: BLE001 - 变体归因失败不影响主流程
            logger.debug(f"Prompt 变体解析失败，按基线处理: {e}")
            return ""

    async def query(
        self,
        question: str,
        session_id: str,
        prompt_variant: str | None = None,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            prompt_variant: AB 变体名（来自 X-Prompt-Variant 请求头）

        Returns:
            str: 完整答案
        """
        try:
            effective_variant = self._resolve_prompt_variant(prompt_variant)
            logger.info(
                f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}"
                + (f" (variant={effective_variant})" if effective_variant else "")
            )

            answer_parts: list[str] = []
            async for event in self.runtime.run(
                question, session_id=session_id, prompt_variant=effective_variant or None
            ):
                if event.type is EventType.TOKEN:
                    answer_parts.append(str(event.payload.get("text", "")))
                elif event.type is EventType.COMPLETE:
                    answer = "".join(answer_parts)
                    logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                    return answer
                elif event.type is EventType.ERROR:
                    raise RuntimeError(event.payload.get("message", "未知错误"))

            # 流意外结束（无 COMPLETE）：返回已收集内容
            return "".join(answer_parts)

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
        prompt_variant: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        旧 SSE chunk 契约（/api/chat_stream 消费）：
            - {"type": "content", "data": 文本块, "node": 节点}
            - {"type": "tool_call", "data": {...}}   ← P1.1 新增（只增不改）
            - {"type": "complete", "prompt_variant": "..."}  ← P5 新增（只增不改）
            - {"type": "error", "data": 错误信息}

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            prompt_variant: AB 变体名（来自 X-Prompt-Variant 请求头）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
        """
        try:
            effective_variant = self._resolve_prompt_variant(prompt_variant)
            logger.info(
                f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}"
                + (f" (variant={effective_variant})" if effective_variant else "")
            )

            async for event in self.runtime.run(
                question, session_id=session_id, prompt_variant=effective_variant or None
            ):
                if event.type is EventType.TOKEN:
                    yield {
                        "type": "content",
                        "data": event.payload.get("text", ""),
                        "node": event.payload.get("node", "unknown"),
                    }
                elif event.type is EventType.TOOL_START:
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": event.payload.get("tool", ""),
                            "status": "start",
                            "input": event.payload.get("args", {}),
                        },
                    }
                elif event.type is EventType.TOOL_END:
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": event.payload.get("tool", ""),
                            "status": event.payload.get("status", "end"),
                            "result_preview": event.payload.get("result_preview", ""),
                        },
                    }
                elif event.type is EventType.COMPLETE:
                    logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
                    complete: dict[str, Any] = {"type": "complete"}
                    # runtime 按实际生效的图回传变体名；基线运行无该字段（只增不改）
                    used_variant = str(event.payload.get("prompt_variant") or "")
                    if used_variant:
                        complete["prompt_variant"] = used_variant
                    yield complete
                elif event.type is EventType.ERROR:
                    message = str(event.payload.get("message", ""))
                    yield {"type": "error", "data": message}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {"type": "error", "data": str(e)}
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        snapshot = self.runtime.snapshot(session_id)
        history: list = snapshot.get("messages", [])
        logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
        return history

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        success = self.runtime.reset(session_id)
        logger.info(f"清空会话: {session_id}, 结果: {success}")
        return success

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
