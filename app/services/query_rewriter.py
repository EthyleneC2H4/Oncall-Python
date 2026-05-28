"""查询改写服务

对用户的原始查询进行改写，提升检索命中率：
- 口语化 → 规范化（如 "服务挂了" → "服务不可用故障排查"）
- 模糊查询 → 具体查询（如 "慢了" → "响应延迟高 性能下降排查"）
- 补充同义词和关键术语
"""

from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config


def _get_rewrite_prompt() -> str:
    """获取查询改写 Prompt，优先从版本化管理器加载"""
    try:
        from app.core.prompt_manager import prompt_manager
        template = prompt_manager.get("rewrite")
        if template:
            return template.content
    except Exception:
        pass

    return """你是一个运维领域的查询改写专家。请将用户的口语化查询改写为更适合检索运维知识库的规范化查询。

改写规则：
1. 保留原始意图，不要改变语义
2. 将口语化/模糊的表述转换为运维领域的专业术语
3. 补充可能的同义词（如：挂了→宕机/不可用/crash）
4. 如果查询已经很规范，保持原样返回
5. 只返回改写后的查询文本，不要解释

示例：
- "服务挂了怎么办" → "服务不可用 宕机 故障排查与恢复方法"
- "CPU跑满了" → "CPU使用率高 CPU负载过高 排查与优化"
- "接口太慢了" → "API响应延迟高 接口慢查询 性能优化"
- "磁盘快满了告警" → "磁盘空间不足告警 磁盘使用率高 清理与扩容"
- "怎么部署这个服务" → "服务部署流程与配置方法"

用户查询: {query}
改写后的查询:"""


REWRITE_PROMPT = _get_rewrite_prompt()


class QueryRewriter:
    """查询改写器"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=0,
            )
        return self._llm

    async def rewrite(self, query: str) -> str:
        """改写用户查询

        Args:
            query: 原始用户查询

        Returns:
            改写后的查询，失败时返回原始查询
        """
        # 短查询或已经很规范的不改写
        if len(query.strip()) <= 4:
            return query

        try:
            result = await self.llm.ainvoke(
                REWRITE_PROMPT.format(query=query)
            )
            rewritten = result.content.strip().strip('"').strip("'")

            if not rewritten or len(rewritten) < 2:
                logger.warning("查询改写结果为空，使用原始查询")
                return query

            logger.info(f"查询改写: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"查询改写失败: {e}，使用原始查询")
            return query


# 全局单例
query_rewriter = QueryRewriter()
