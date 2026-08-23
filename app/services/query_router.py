"""查询意图分类与路由

根据用户查询自动分类意图并路由到不同的处理策略：
- DIAGNOSTIC: 故障诊断类 → 优先 KG + RAG + AIOps
- KNOWLEDGE: 知识查询类 → RAG 检索（含 HyDE）
- CHITCHAT: 闲聊类 → 直接 LLM 回答
- COMPLEX: 复杂分析类 → 多步检索 + HyDE
"""

import json
import re

from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory

# 意图分类关键词规则（快速规则优先，LLM 兜底）
_DIAGNOSTIC_KEYWORDS = [
    "告警",
    "alert",
    "报警",
    "异常",
    "故障",
    "宕机",
    "crash",
    "oom",
    "cpu高",
    "内存高",
    "磁盘满",
    "响应慢",
    "超时",
    "不可用",
    "错误率",
    "down",
    "error",
    "fail",
    "timeout",
    "unavailable",
    "high cpu",
    "high memory",
    "disk full",
    "slow response",
]

_KNOWLEDGE_KEYWORDS = [
    "怎么",
    "如何",
    "什么是",
    "步骤",
    "方法",
    "教程",
    "文档",
    "how to",
    "what is",
    "guide",
    "tutorial",
    "best practice",
    "排查",
    "处理",
    "解决",
    "配置",
    "部署",
    "安装",
]

_CHITCHAT_KEYWORDS = [
    "你好",
    "hello",
    "hi",
    "谢谢",
    "thanks",
    "再见",
    "bye",
    "你是谁",
    "who are you",
    "帮助",
    "help",
]


class QueryRouter:
    """查询意图分类与路由"""

    INTENT_PROMPT = """分析以下用户查询的意图类型，返回 JSON 格式结果。

意图类型定义：
- DIAGNOSTIC: 故障诊断类（包含告警、异常、错误、性能问题等关键词）
- KNOWLEDGE: 知识查询类（怎么做、什么是、如何处理、最佳实践等）
- CHITCHAT: 闲聊类（问候、感谢、身份询问等）
- COMPLEX: 复杂分析类（需要多步推理、涉及多个系统/服务的综合分析）

用户查询: {query}

请返回如下 JSON 格式（不要包含其他内容）：
{{"intent": "DIAGNOSTIC", "keywords": ["关键词1", "关键词2"], "confidence": 0.9}}"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.create_chat_model(
                streaming=False,
                model=config.rag_model,
                temperature=0,
            )
        return self._llm

    def route_by_rules(self, query: str) -> tuple[str, list[str]]:
        """基于规则的快速意图分类（无 LLM 开销）

        Args:
            query: 用户查询

        Returns:
            (intent, keywords) 意图类型和提取的关键词
        """
        query_lower = query.lower()
        matched_keywords = []

        # 检查诊断类
        for kw in _DIAGNOSTIC_KEYWORDS:
            if kw in query_lower:
                matched_keywords.append(kw)
        if matched_keywords:
            return "DIAGNOSTIC", matched_keywords

        # 检查知识类
        for kw in _KNOWLEDGE_KEYWORDS:
            if kw in query_lower:
                matched_keywords.append(kw)
        if matched_keywords:
            return "KNOWLEDGE", matched_keywords

        # 检查闲聊类
        for kw in _CHITCHAT_KEYWORDS:
            if kw in query_lower:
                return "CHITCHAT", [kw]

        # 默认为知识类
        return "KNOWLEDGE", []

    async def route(self, query: str) -> tuple[str, list[str]]:
        """意图分类（规则优先 + LLM 兜底）

        Args:
            query: 用户查询

        Returns:
            (intent, keywords) 意图类型和提取的关键词
        """
        # 先尝试规则分类
        intent, keywords = self.route_by_rules(query)

        # 如果规则分类有明确结果且关键词多，直接返回
        if keywords and len(keywords) >= 1:
            logger.info(f"查询路由（规则）: intent={intent}, keywords={keywords}")
            return intent, keywords

        # LLM 兜底分类
        try:
            result = await self.llm.ainvoke(self.INTENT_PROMPT.format(query=query))
            content = result.content.strip()

            # 提取 JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                intent = parsed.get("intent", "KNOWLEDGE")
                keywords = parsed.get("keywords", [])
                logger.info(f"查询路由（LLM）: intent={intent}, keywords={keywords}")
                return intent, keywords
        except Exception as e:
            logger.warning(f"LLM 意图分类失败: {e}, 使用规则结果")

        return intent, keywords

    def get_retrieval_strategy(self, intent: str) -> dict:
        """根据意图返回检索策略配置

        Args:
            intent: 意图类型

        Returns:
            dict: 检索策略配置
        """
        strategies = {
            "DIAGNOSTIC": {
                "use_kg": True,
                "use_rag": True,
                "use_hyde": False,
                "rag_top_k": config.rag_top_k,
            },
            "KNOWLEDGE": {
                "use_kg": False,
                "use_rag": True,
                "use_hyde": True,
                "rag_top_k": config.rag_top_k,
            },
            "CHITCHAT": {
                "use_kg": False,
                "use_rag": False,
                "use_hyde": False,
                "rag_top_k": 0,
            },
            "COMPLEX": {
                "use_kg": True,
                "use_rag": True,
                "use_hyde": True,
                "rag_top_k": config.rag_top_k + 2,
            },
        }
        return strategies.get(intent, strategies["KNOWLEDGE"])


# 全局单例
query_router = QueryRouter()
