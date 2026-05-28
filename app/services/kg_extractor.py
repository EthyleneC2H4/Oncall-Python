"""知识图谱自动抽取 - 从运维文档中自动抽取实体和关系

使用 LLM 从文档中提取三元组（entity-relation-entity），
替代手动硬编码的知识图谱构建方式。
"""

import json
import re
from typing import Any

from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config


class Triple(BaseModel):
    """知识图谱三元组"""
    head: str = Field(description="头实体")
    head_type: str = Field(description="头实体类型: alert | root_cause | action | tool | metric")
    relation: str = Field(description="关系类型: CAUSED_BY | RESOLVED_BY | MAY_TRIGGER | USES | INDICATES")
    tail: str = Field(description="尾实体")
    tail_type: str = Field(description="尾实体类型")


class IncidentRecord(BaseModel):
    """已解决的故障事件记录"""
    incident_id: str = Field(description="事件ID")
    alert_type: str = Field(description="告警类型")
    root_cause: str = Field(default="", description="根因")
    resolution: str = Field(default="", description="解决方案")
    cascade_alerts: list[str] = Field(default_factory=list, description="级联告警列表")
    timestamp: str = Field(default="", description="事件时间")


def _get_extract_prompt() -> str:
    """获取 KG 抽取 Prompt，优先从版本化管理器加载"""
    try:
        from app.core.prompt_manager import prompt_manager
        template = prompt_manager.get("extract")
        if template:
            return template.content
    except Exception:
        pass

    return """从以下运维文档中抽取实体和关系三元组。

## 实体类型
- alert: 告警类型（如 CPU使用率过高、内存不足、服务不可用）
- root_cause: 根因（如 内存泄漏、死循环、流量突增）
- action: 处置动作（如 重启服务、扩容、回滚代码）
- tool: 工具（如 top命令、jmap、日志查询）
- metric: 监控指标（如 CPU使用率、内存使用率、响应时间）

## 关系类型
- CAUSED_BY: 告警由根因引起
- RESOLVED_BY: 根因通过某个动作解决
- MAY_TRIGGER: 告警可能触发另一个告警（级联）
- USES: 动作使用某个工具
- INDICATES: 指标异常表明某个告警

## 文档内容
{document}

## 输出格式
请输出 JSON 格式，不要包含其他内容：
{{"triples": [
    {{"head": "CPU使用率过高", "head_type": "alert", "relation": "CAUSED_BY", "tail": "死循环", "tail_type": "root_cause"}},
    {{"head": "死循环", "head_type": "root_cause", "relation": "RESOLVED_BY", "tail": "重启服务", "tail_type": "action"}}
]}}

注意：
- 只抽取文档中明确提到的实体和关系，不要推测
- 实体名称使用简洁的中文
- 每个三元组必须包含所有字段
"""


EXTRACT_PROMPT = _get_extract_prompt()


class KGExtractor:
    """从运维文档自动抽取知识图谱三元组"""

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

    async def extract_from_text(self, document: str) -> list[Triple]:
        """从单个文档抽取三元组

        Args:
            document: 文档文本内容

        Returns:
            抽取到的三元组列表
        """
        try:
            # 截断过长文档
            if len(document) > 3000:
                document = document[:3000] + "\n...(文档已截断)"

            result = await self.llm.ainvoke(
                EXTRACT_PROMPT.format(document=document)
            )
            content = result.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.warning("LLM 未返回有效 JSON")
                return []

            parsed = json.loads(json_match.group())
            triples = []
            for t in parsed.get("triples", []):
                try:
                    triples.append(Triple(**t))
                except Exception:
                    logger.warning(f"跳过无效三元组: {t}")

            logger.info(f"从文档抽取到 {len(triples)} 个三元组")
            return triples

        except Exception as e:
            logger.error(f"三元组抽取失败: {e}")
            return []

    async def extract_from_docs(self, documents: list[str]) -> list[Triple]:
        """批量从文档抽取三元组，自动去重

        Args:
            documents: 文档文本列表

        Returns:
            去重后的三元组列表
        """
        all_triples = []
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            triples = await self.extract_from_text(doc)
            all_triples.extend(triples)

        # 去重
        deduplicated = self._deduplicate(all_triples)
        logger.info(f"总计抽取 {len(all_triples)} 个三元组，去重后 {len(deduplicated)} 个")
        return deduplicated

    def _deduplicate(self, triples: list[Triple]) -> list[Triple]:
        """三元组去重（基于 head+relation+tail 的组合）"""
        seen = set()
        result = []
        for t in triples:
            key = (t.head.strip(), t.relation, t.tail.strip())
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result


# 全局单例
kg_extractor = KGExtractor()
