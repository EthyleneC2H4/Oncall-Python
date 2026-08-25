"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理

LLM 接入：OpenRouter（OpenAI 兼容模式）
向量化/重排：本地模型（sentence-transformers / FlagEmbedding），零 API 成本
"""

from typing import Any

from pydantic import Field
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

    # 安全配置（P3）
    auth_enabled: bool = False  # X-API-Key 静态密钥鉴权；默认关（本地开发零负担）
    auth_api_key: str = ""  # 鉴权密钥；enabled 时必须非空，仅从 .env 注入
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    # 通配 origin 与 credentials 是无效组合（浏览器规范禁止），引擎层强制互斥；
    # 显式域名列表时可安全开启
    cors_allow_credentials: bool = False

    # 待审动作（P3 高风险工具确认门）
    pending_actions_db_path: str = "data/pending_actions.db"  # sqlite 存储路径
    pending_action_ttl_seconds: float = 900.0  # 待审动作过期时间（15 分钟）

    # 工具调用痕迹（P4 BFCL 式评测数据源）
    tool_trace_enabled: bool = True  # 关闭后 guard 不再写 data/traces/
    traces_dir: str = "data/traces"

    # OpenRouter 配置（OpenAI 兼容模式）
    openrouter_api_key: str = ""  # 默认空字符串，实际使用需从环境变量 OPENROUTER_API_KEY 加载
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "nvidia/nemotron-3.5-lightning:free"  # 对话/规划主模型（OpenRouter slug，免费档）

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

    # 长期记忆配置（P2）
    # 数值项用 Field 约束边界：env 误配（如 nan/负值）在启动即失败，
    # 而非静默毒化打分（λ=nan 会让全部得分变 NaN、召回永久返回空）
    memory_enabled: bool = True  # 总开关；关闭时所有记忆读写为无副作用空操作
    memory_db_path: str = "data/memory.db"  # sqlite 存储路径（WAL 模式）
    memory_recall_k: int = Field(default=5, ge=1)  # 单次召回上限
    memory_min_importance: float = Field(default=0.2, ge=0.0, le=1.0)  # 召回的重要性下限
    memory_decay_lambda: float = Field(default=0.05, ge=0.0)  # 新近度指数衰减率 λ（按天）
    memory_weight_relevance: float = Field(default=0.6, ge=0.0, le=1.0)  # 权重：向量相关性
    memory_weight_importance: float = Field(default=0.25, ge=0.0, le=1.0)  # 权重：重要性
    memory_weight_recency: float = Field(default=0.15, ge=0.0, le=1.0)  # 权重：新近度
    memory_consolidate_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0
    )  # 情景→语义巩固的余弦相似度阈值

    # 上下文引擎配置（P2）
    context_token_budget: int = 6000  # 注入 LLM 的上下文总预算（近似估算）
    context_history_budget: int = 2400  # ReAct 对话历史 token 预算（替换旧硬编码条数截断）

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
