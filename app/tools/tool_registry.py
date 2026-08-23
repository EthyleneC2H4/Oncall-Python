"""工具权限与风险分级注册中心

每个工具声明风险等级和参数 Schema，
调用前进行权限校验和参数验证。
"""

from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, Field

from app.core.audit import audit_logger


class RiskLevel(StrEnum):
    """工具风险等级"""

    READ_ONLY = "read_only"  # 只读操作，直接执行
    READ_AUDIT = "read_audit"  # 只读但需审计
    WRITE_LOW = "write_low"  # 低风险写操作
    WRITE_HIGH_RISK = "write_high_risk"  # 高风险写操作，需人工确认


class ToolMeta(BaseModel):
    """工具元信息"""

    name: str
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    description: str = ""
    param_schema: dict = Field(default_factory=dict)
    requires_confirmation: bool = False
    audit: bool = False


# 预定义工具注册表
_TOOL_REGISTRY: dict[str, ToolMeta] = {
    # 本地工具
    "get_current_time": ToolMeta(
        name="get_current_time",
        risk_level=RiskLevel.READ_ONLY,
        description="获取当前时间戳",
    ),
    "retrieve_knowledge": ToolMeta(
        name="retrieve_knowledge",
        risk_level=RiskLevel.READ_ONLY,
        description="知识库检索",
    ),
    "query_alert_graph": ToolMeta(
        name="query_alert_graph",
        risk_level=RiskLevel.READ_ONLY,
        description="知识图谱告警分析",
    ),
    "predict_alert_cascade": ToolMeta(
        name="predict_alert_cascade",
        risk_level=RiskLevel.READ_ONLY,
        description="告警级联预测",
    ),
    # MCP: CLS 工具
    "search_log": ToolMeta(
        name="search_log",
        risk_level=RiskLevel.READ_AUDIT,
        description="日志查询",
        audit=True,
        param_schema={
            "topic_id": {"type": "string", "required": True},
            "start_time": {"type": "integer", "required": True},
            "end_time": {"type": "integer", "required": True},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
    ),
    "get_topic_info_by_name": ToolMeta(
        name="get_topic_info_by_name",
        risk_level=RiskLevel.READ_ONLY,
        description="获取日志主题信息",
    ),
    "describe_topic": ToolMeta(
        name="describe_topic",
        risk_level=RiskLevel.READ_ONLY,
        description="描述日志主题",
    ),
    "list_topics": ToolMeta(
        name="list_topics",
        risk_level=RiskLevel.READ_ONLY,
        description="列出日志主题",
    ),
    "get_histograms": ToolMeta(
        name="get_histograms",
        risk_level=RiskLevel.READ_ONLY,
        description="获取日志直方图",
    ),
    # MCP: Monitor 工具
    "query_cpu_metrics": ToolMeta(
        name="query_cpu_metrics",
        risk_level=RiskLevel.READ_AUDIT,
        description="查询 CPU 指标",
        audit=True,
    ),
    "query_memory_metrics": ToolMeta(
        name="query_memory_metrics",
        risk_level=RiskLevel.READ_AUDIT,
        description="查询内存指标",
        audit=True,
    ),
    # 未来高风险工具（预留）
    "restart_instance": ToolMeta(
        name="restart_instance",
        risk_level=RiskLevel.WRITE_HIGH_RISK,
        description="重启实例",
        requires_confirmation=True,
        audit=True,
    ),
    "scale_out": ToolMeta(
        name="scale_out",
        risk_level=RiskLevel.WRITE_HIGH_RISK,
        description="实例扩容",
        requires_confirmation=True,
        audit=True,
    ),
}


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._registry = dict(_TOOL_REGISTRY)

    def register(self, tool_meta: ToolMeta):
        """注册工具"""
        self._registry[tool_meta.name] = tool_meta
        logger.debug(f"注册工具: {tool_meta.name} (risk={tool_meta.risk_level})")

    def get(self, name: str) -> ToolMeta | None:
        """获取工具元信息"""
        meta: ToolMeta | None = self._registry.get(name)
        return meta

    def check_permission(self, tool_name: str) -> tuple[bool, str]:
        """检查工具调用权限

        Returns:
            (allowed, reason)
        """
        meta = self.get(tool_name)

        if meta is None:
            # 未注册的工具默认允许（兼容动态 MCP 工具）
            logger.debug(f"工具 '{tool_name}' 未在注册表中，默认允许")
            return True, ""

        if meta.requires_confirmation:
            return False, f"工具 '{tool_name}' 为高风险操作，需要人工确认"

        return True, ""

    def validate_params(self, tool_name: str, params: dict) -> tuple[bool, str]:
        """校验工具参数

        Returns:
            (valid, error_message)
        """
        meta = self.get(tool_name)
        if not meta or not meta.param_schema:
            return True, ""

        # 检查必需参数
        for param_name, schema in meta.param_schema.items():
            if schema.get("required", False) and param_name not in params:
                return False, f"缺少必需参数: {param_name}"

            if param_name in params:
                value = params[param_name]
                # 类型检查
                expected_type = schema.get("type", "")
                if expected_type == "integer" and not isinstance(value, int):
                    return False, f"参数 '{param_name}' 应为整数"
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"参数 '{param_name}' 应为字符串"

                # 范围检查
                if "minimum" in schema and isinstance(value, (int, float)):
                    if value < schema["minimum"]:
                        return False, f"参数 '{param_name}' 不能小于 {schema['minimum']}"
                if "maximum" in schema and isinstance(value, (int, float)):
                    if value > schema["maximum"]:
                        return False, f"参数 '{param_name}' 不能大于 {schema['maximum']}"

        return True, ""

    def should_audit(self, tool_name: str) -> bool:
        """是否需要审计记录"""
        meta = self.get(tool_name)
        return meta.audit if meta else False

    def audit_call(
        self,
        tool_name: str,
        params: dict | None = None,
        result_status: str = "success",
        request_id: str = "",
        latency_ms: float = 0,
        error: str | None = None,
    ):
        """记录工具调用审计"""
        if self.should_audit(tool_name):
            audit_logger.log_tool_call(
                request_id=request_id,
                tool_name=tool_name,
                params=params,
                result_status=result_status,
                latency_ms=latency_ms,
                error=error,
            )

    def list_tools(self) -> list[dict]:
        """列出所有已注册工具"""
        return [
            {
                "name": meta.name,
                "risk_level": meta.risk_level.value,
                "description": meta.description,
                "requires_confirmation": meta.requires_confirmation,
                "audit": meta.audit,
            }
            for meta in self._registry.values()
        ]


# 全局单例
tool_registry = ToolRegistry()
