"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理

LLM 接入：OpenRouter（OpenAI 兼容模式）
向量化/重排：本地模型（sentence-transformers / FlagEmbedding），零 API 成本
"""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "2.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # OpenRouter 配置（OpenAI 兼容模式）
    openrouter_api_key: str = ""  # 默认空字符串，实际使用需从环境变量 OPENROUTER_API_KEY 加载
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3.5-lightning"

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "nvidia/nemotron-3.5-lightning"  # 对话/规划主模型（OpenRouter slug）

    # 本地向量化配置（sentence-transformers）
    embedding_model: str = "BAAI/bge-large-zh-v1.5"  # 中文优化，1024 维，与原 Milvus schema 兼容
    embedding_dimensions: int = 1024
    embedding_device: str = ""  # 留空自动选择：mps（Apple Silicon）> cpu

    # 本地重排配置（FlagEmbedding cross-encoder）
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 降级策略配置
    llm_backup_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"  # 弱模型层（OpenRouter slug）
    llm_timeout_default: float = 30.0  # LLM 默认超时（秒）
    llm_timeout_simple: float = 10.0  # 简单任务超时（改写/路由）
    llm_timeout_complex: float = 30.0  # 复杂任务超时（规划/报告）
    circuit_failure_threshold: int = 5  # 熔断触发的连续失败次数
    circuit_cooldown_seconds: float = 60.0  # 熔断冷却时间（秒）
    health_probe_interval: float = 30.0  # 健康探针间隔（秒）
    step_timeout_seconds: float = 60.0  # Agent 单步超时（秒）
    workflow_timeout_seconds: float = 180.0  # Agent 整体工作流超时（秒）

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }


# 全局配置实例
config = Settings()
