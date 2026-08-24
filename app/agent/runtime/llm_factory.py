"""运行时 LLM 分层工厂

strong / weak 双层模型策略（借鉴 Cortex cortex/llm/factory.py 的分层思想）：

- strong 层：规划、执行、报告生成等复杂推理任务，使用主模型
  （config.rag_model，Nemotron 3.5 Lightning），允许长思考。
- weak 层：意图路由、查询改写、打分等简单任务，使用备用轻量模型
  （config.llm_backup_model），追求低延迟低成本。

实例缓存与降级链由 app.core.llm_factory.LLMFactory 统一提供，
本模块只封装「按任务层级选模」的策略语义，避免调用方散落硬编码模型 slug。
"""

from langchain_openai import ChatOpenAI

from app.core.llm_factory import LLMFactory


class TieredLLM:
    """strong / weak 双层模型选择策略"""

    @staticmethod
    def strong(temperature: float = 0.0) -> ChatOpenAI:
        """强模型层：规划 / 报告等复杂任务"""
        return LLMFactory.strong(temperature)

    @staticmethod
    def weak(temperature: float = 0.0) -> ChatOpenAI:
        """弱模型层：路由 / 改写 / 打分等简单任务（快、省）"""
        return LLMFactory.weak(temperature)


# 模块级单例，供各运行时 / 节点注入使用
tiered_llm = TieredLLM()
