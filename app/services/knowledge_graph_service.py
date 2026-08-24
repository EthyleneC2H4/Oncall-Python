"""运维知识图谱服务 - 基于 NetworkX 的轻量级知识图谱

提供告警-根因-处置动作之间的关联推理能力，
支持告警级联链路分析和多跳关联查询。
"""

import os
from typing import Any

import networkx as nx
from loguru import logger

# 持久化文件路径
KG_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "kg_data.json")


class KnowledgeGraphService:
    """运维知识图谱服务"""

    # 口语关键词 → 告警节点（图检索通道与模糊匹配共用）
    ALERT_KEYWORDS: dict[str, str] = {
        "cpu": "HighCPUUsage",
        "内存": "HighMemoryUsage",
        "memory": "HighMemoryUsage",
        "mem": "HighMemoryUsage",
        "磁盘": "HighDiskUsage",
        "disk": "HighDiskUsage",
        "响应": "SlowResponse",
        "slow": "SlowResponse",
        "慢": "SlowResponse",
        "不可用": "ServiceUnavailable",
        "unavailable": "ServiceUnavailable",
        "崩溃": "ServiceUnavailable",
        "crash": "ServiceUnavailable",
        "oom": "OOMError",
        "错误率": "HighErrorRate",
        "error": "HighErrorRate",
        "gc": "FrequentGC",
        "健康检查": "HealthCheckFailed",
    }

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_graph()
        logger.info(
            f"知识图谱初始化完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边"
        )

    def _build_default_graph(self):
        """从运维文档知识构建默认图谱"""

        # ========== 告警类型节点 ==========
        alerts = {
            "HighCPUUsage": {
                "level": "critical",
                "label": "CPU使用率过高",
                "threshold": "80%持续5分钟",
            },
            "HighMemoryUsage": {
                "level": "critical",
                "label": "内存使用率过高",
                "threshold": "85%持续5分钟",
            },
            "HighDiskUsage": {
                "level": "warning",
                "label": "磁盘使用率过高",
                "threshold": "80%持续5分钟",
            },
            "SlowResponse": {
                "level": "warning",
                "label": "响应变慢",
                "threshold": "P99>3s持续5分钟",
            },
            "ServiceUnavailable": {
                "level": "critical",
                "label": "服务不可用",
                "threshold": "健康检查失败或错误率>50%",
            },
            "HighErrorRate": {
                "level": "warning",
                "label": "高错误率",
                "threshold": "错误率异常升高",
            },
            "DatabaseSlowQuery": {
                "level": "warning",
                "label": "数据库慢查询",
                "threshold": "慢查询数异常",
            },
            "OOMError": {"level": "critical", "label": "内存溢出", "threshold": "OOM Killer触发"},
            "HealthCheckFailed": {
                "level": "critical",
                "label": "健康检查失败",
                "threshold": "连续检查失败",
            },
            "FrequentGC": {"level": "warning", "label": "GC频繁", "threshold": "GC频率异常"},
            "DiskIOHigh": {
                "level": "warning",
                "label": "磁盘IO过高",
                "threshold": "IO等待时间过长",
            },
        }

        for name, attrs in alerts.items():
            self.graph.add_node(name, type="alert", **attrs)

        # ========== 根因节点 ==========
        root_causes = {
            "InfiniteLoop": {"label": "死循环/无限递归", "category": "code"},
            "TrafficSurge": {"label": "流量突增", "category": "traffic"},
            "ScheduledTaskOverlap": {"label": "定时任务重叠执行", "category": "config"},
            "DBSlowQuery": {"label": "数据库慢查询", "category": "database"},
            "MemoryLeak": {"label": "内存泄漏", "category": "code"},
            "CacheMisconfigured": {"label": "缓存配置不当", "category": "config"},
            "LargeFileProcessing": {"label": "大文件/大量数据处理", "category": "workload"},
            "JVMConfigIssue": {"label": "JVM参数配置不当", "category": "config"},
            "LogFileTooLarge": {"label": "日志文件过大", "category": "ops"},
            "TempFileAccumulation": {"label": "临时文件堆积", "category": "ops"},
            "DataFileGrowth": {"label": "数据文件快速增长", "category": "data"},
            "DockerImageBloat": {"label": "Docker镜像/容器占用空间", "category": "ops"},
            "ExternalAPITimeout": {"label": "外部API调用超时", "category": "dependency"},
            "CacheInvalidation": {"label": "缓存失效/穿透", "category": "cache"},
            "NetworkIssue": {"label": "网络问题", "category": "infra"},
            "AppCrash": {"label": "应用崩溃/启动失败", "category": "code"},
            "DBConnectionFailure": {"label": "数据库连接失败", "category": "database"},
            "DependencyFailure": {"label": "依赖服务故障", "category": "dependency"},
            "ConfigError": {"label": "配置错误", "category": "config"},
            "ResourceExhaustion": {"label": "资源耗尽", "category": "infra"},
        }

        for name, attrs in root_causes.items():
            self.graph.add_node(name, type="root_cause", **attrs)

        # ========== 处置动作节点 ==========
        actions = {
            "ScaleOut": {"label": "水平扩容", "urgency": "immediate"},
            "RateLimit": {"label": "启用限流", "urgency": "immediate"},
            "RestartInstance": {"label": "重启实例", "urgency": "immediate"},
            "Rollback": {"label": "代码回滚", "urgency": "short_term"},
            "AnalyzeLogs": {"label": "分析日志定位根因", "urgency": "short_term"},
            "OptimizeSQL": {"label": "优化慢SQL", "urgency": "long_term"},
            "FixMemoryLeak": {"label": "修复内存泄漏", "urgency": "long_term"},
            "AdjustJVMParams": {"label": "调整JVM参数", "urgency": "short_term"},
            "CleanupLogs": {"label": "清理日志文件", "urgency": "immediate"},
            "CleanupTempFiles": {"label": "清理临时文件", "urgency": "immediate"},
            "DiskExpansion": {"label": "磁盘扩容", "urgency": "immediate"},
            "SetupLogRotation": {"label": "配置日志轮转", "urgency": "long_term"},
            "DegradeNonCritical": {"label": "降级非关键功能", "urgency": "immediate"},
            "SwitchToBackup": {"label": "切换备用服务", "urgency": "immediate"},
            "OptimizeCode": {"label": "代码性能优化", "urgency": "long_term"},
            "AddCache": {"label": "增加缓存", "urgency": "long_term"},
            "ClearCache": {"label": "手动清理缓存", "urgency": "short_term"},
            "DumpMemory": {"label": "导出内存快照分析", "urgency": "short_term"},
        }

        for name, attrs in actions.items():
            self.graph.add_node(name, type="action", **attrs)

        # ========== 工具节点 ==========
        tools = {
            "top_command": {"label": "top命令", "usage": "查看进程CPU占用"},
            "jmap_command": {"label": "jmap命令", "usage": "JVM内存分析"},
            "jstat_command": {"label": "jstat命令", "usage": "JVM GC统计"},
            "df_command": {"label": "df命令", "usage": "查看磁盘使用"},
            "du_command": {"label": "du命令", "usage": "查看目录大小"},
            "query_logs_tool": {"label": "日志查询工具", "usage": "MCP CLS日志查询"},
            "query_metrics_tool": {"label": "指标查询工具", "usage": "MCP Monitor指标查询"},
        }

        for name, attrs in tools.items():
            self.graph.add_node(name, type="tool", **attrs)

        # ========== 关系边 ==========

        # --- 告警 CAUSED_BY 根因 ---
        caused_by = [
            ("HighCPUUsage", "InfiniteLoop"),
            ("HighCPUUsage", "TrafficSurge"),
            ("HighCPUUsage", "ScheduledTaskOverlap"),
            ("HighCPUUsage", "DBSlowQuery"),
            ("HighMemoryUsage", "MemoryLeak"),
            ("HighMemoryUsage", "TrafficSurge"),
            ("HighMemoryUsage", "CacheMisconfigured"),
            ("HighMemoryUsage", "LargeFileProcessing"),
            ("HighMemoryUsage", "JVMConfigIssue"),
            ("HighDiskUsage", "LogFileTooLarge"),
            ("HighDiskUsage", "TempFileAccumulation"),
            ("HighDiskUsage", "DataFileGrowth"),
            ("HighDiskUsage", "DockerImageBloat"),
            ("SlowResponse", "DBSlowQuery"),
            ("SlowResponse", "ExternalAPITimeout"),
            ("SlowResponse", "CacheInvalidation"),
            ("SlowResponse", "TrafficSurge"),
            ("SlowResponse", "NetworkIssue"),
            ("SlowResponse", "MemoryLeak"),
            ("ServiceUnavailable", "AppCrash"),
            ("ServiceUnavailable", "DBConnectionFailure"),
            ("ServiceUnavailable", "DependencyFailure"),
            ("ServiceUnavailable", "ConfigError"),
            ("ServiceUnavailable", "ResourceExhaustion"),
            ("ServiceUnavailable", "NetworkIssue"),
        ]

        for alert, cause in caused_by:
            self.graph.add_edge(alert, cause, relation="CAUSED_BY")

        # --- 根因 RESOLVED_BY 处置动作 ---
        resolved_by = [
            ("InfiniteLoop", "RestartInstance"),
            ("InfiniteLoop", "Rollback"),
            ("InfiniteLoop", "OptimizeCode"),
            ("TrafficSurge", "ScaleOut"),
            ("TrafficSurge", "RateLimit"),
            ("ScheduledTaskOverlap", "RestartInstance"),
            ("DBSlowQuery", "OptimizeSQL"),
            ("DBSlowQuery", "AnalyzeLogs"),
            ("MemoryLeak", "RestartInstance"),
            ("MemoryLeak", "DumpMemory"),
            ("MemoryLeak", "FixMemoryLeak"),
            ("CacheMisconfigured", "ClearCache"),
            ("CacheMisconfigured", "AddCache"),
            ("JVMConfigIssue", "AdjustJVMParams"),
            ("LogFileTooLarge", "CleanupLogs"),
            ("LogFileTooLarge", "SetupLogRotation"),
            ("TempFileAccumulation", "CleanupTempFiles"),
            ("DataFileGrowth", "DiskExpansion"),
            ("ExternalAPITimeout", "DegradeNonCritical"),
            ("CacheInvalidation", "AddCache"),
            ("CacheInvalidation", "ClearCache"),
            ("AppCrash", "RestartInstance"),
            ("AppCrash", "Rollback"),
            ("AppCrash", "AnalyzeLogs"),
            ("DBConnectionFailure", "RestartInstance"),
            ("DBConnectionFailure", "AnalyzeLogs"),
            ("DependencyFailure", "DegradeNonCritical"),
            ("DependencyFailure", "SwitchToBackup"),
            ("ConfigError", "Rollback"),
            ("ResourceExhaustion", "ScaleOut"),
            ("ResourceExhaustion", "DiskExpansion"),
            ("NetworkIssue", "AnalyzeLogs"),
            ("LargeFileProcessing", "ScaleOut"),
            ("DockerImageBloat", "CleanupTempFiles"),
        ]

        for cause, action in resolved_by:
            self.graph.add_edge(cause, action, relation="RESOLVED_BY")

        # --- 告警 MAY_TRIGGER 告警（级联关系，核心亮点）---
        may_trigger = [
            ("HighCPUUsage", "SlowResponse", "CPU过高导致处理变慢"),
            ("HighCPUUsage", "HighErrorRate", "CPU过高导致请求失败"),
            ("HighMemoryUsage", "FrequentGC", "内存压力触发频繁GC"),
            ("HighMemoryUsage", "OOMError", "内存持续升高触发OOM"),
            ("HighMemoryUsage", "HighCPUUsage", "频繁GC消耗CPU"),
            ("FrequentGC", "SlowResponse", "GC暂停导致响应变慢"),
            ("FrequentGC", "HighCPUUsage", "GC消耗大量CPU"),
            ("OOMError", "ServiceUnavailable", "OOM导致进程被杀"),
            ("SlowResponse", "HighErrorRate", "响应超时导致错误率上升"),
            ("SlowResponse", "ServiceUnavailable", "长时间慢响应导致服务崩溃"),
            ("HighErrorRate", "ServiceUnavailable", "错误率过高导致服务不可用"),
            ("HighDiskUsage", "DiskIOHigh", "磁盘满导致IO性能下降"),
            ("HighDiskUsage", "ServiceUnavailable", "磁盘满导致无法写入日志/数据"),
            ("DatabaseSlowQuery", "SlowResponse", "慢查询直接导致响应变慢"),
            ("DatabaseSlowQuery", "HighCPUUsage", "慢查询消耗大量CPU"),
        ]

        for src, dst, reason in may_trigger:
            self.graph.add_edge(src, dst, relation="MAY_TRIGGER", reason=reason)

        # --- 处置动作 USES 工具 ---
        uses_tool = [
            ("AnalyzeLogs", "query_logs_tool"),
            ("AnalyzeLogs", "query_metrics_tool"),
            ("DumpMemory", "jmap_command"),
            ("AdjustJVMParams", "jstat_command"),
            ("CleanupLogs", "du_command"),
            ("CleanupLogs", "df_command"),
            ("CleanupTempFiles", "du_command"),
        ]

        for action, tool in uses_tool:
            self.graph.add_edge(action, tool, relation="USES")

    def get_alert_analysis(self, alert_name: str) -> dict[str, Any]:
        """获取告警的完整分析（根因 + 处置 + 级联链路）

        Args:
            alert_name: 告警名称（如 HighCPUUsage）

        Returns:
            包含根因、处置动作和级联链路的完整分析
        """
        if alert_name not in self.graph:
            # 尝试模糊匹配
            matched = self._fuzzy_match_alert(alert_name)
            if not matched:
                return {"error": "未找到匹配的告警类型"}
            alert_name = matched

        node_data = self.graph.nodes[alert_name]
        result = {
            "alert": alert_name,
            "label": node_data.get("label", ""),
            "level": node_data.get("level", ""),
            "root_causes": [],
            "cascade_chain": [],
            "recommended_actions": [],
        }

        # 1. 获取根因
        for _, target, data in self.graph.out_edges(alert_name, data=True):
            if data.get("relation") == "CAUSED_BY":
                cause_data = self.graph.nodes[target]
                actions = []
                for _, action_target, action_data in self.graph.out_edges(target, data=True):
                    if action_data.get("relation") == "RESOLVED_BY":
                        action_node = self.graph.nodes[action_target]
                        actions.append(
                            {
                                "name": action_target,
                                "label": action_node.get("label", ""),
                                "urgency": action_node.get("urgency", ""),
                            }
                        )
                result["root_causes"].append(
                    {
                        "name": target,
                        "label": cause_data.get("label", ""),
                        "category": cause_data.get("category", ""),
                        "actions": actions,
                    }
                )

        # 2. 获取级联链路（BFS，最多3层）
        result["cascade_chain"] = self._get_cascade_chain(alert_name, max_depth=3)

        # 3. 汇总推荐动作（按紧急度排序）
        seen_actions = set()
        urgency_order = {"immediate": 0, "short_term": 1, "long_term": 2}
        for cause in result["root_causes"]:
            for action in cause["actions"]:
                if action["name"] not in seen_actions:
                    seen_actions.add(action["name"])
                    result["recommended_actions"].append(action)
        result["recommended_actions"].sort(key=lambda a: urgency_order.get(a["urgency"], 99))

        return result

    def _get_cascade_chain(self, alert_name: str, max_depth: int = 3) -> list[dict]:
        """获取告警级联链路（BFS）"""
        chains = []
        visited = {alert_name}
        queue = [(alert_name, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for _, target, data in self.graph.out_edges(current, data=True):
                if data.get("relation") == "MAY_TRIGGER" and target not in visited:
                    visited.add(target)
                    target_data = self.graph.nodes[target]
                    chain_item = {
                        "from": current,
                        "from_label": self.graph.nodes[current].get("label", ""),
                        "to": target,
                        "to_label": target_data.get("label", ""),
                        "reason": data.get("reason", ""),
                        "depth": depth + 1,
                    }
                    chains.append(chain_item)
                    queue.append((target, depth + 1))

        return chains

    def get_cascade_prediction(self, alert_name: str) -> str:
        """获取告警级联预测的可读文本"""
        if alert_name not in self.graph:
            matched = self._fuzzy_match_alert(alert_name)
            if not matched:
                return "未找到匹配的告警类型"
            alert_name = matched

        chains = self._get_cascade_chain(alert_name, max_depth=3)
        if not chains:
            return f"{alert_name} 无已知级联风险"

        lines = [f"⚠️ {self.graph.nodes[alert_name].get('label', alert_name)} 可能引发的级联告警："]
        for item in chains:
            indent = "  " * item["depth"]
            lines.append(f"{indent}→ {item['to_label']}（{item['reason']}）")

        return "\n".join(lines)

    def format_analysis_context(self, alert_name: str) -> str:
        """将图谱分析结果格式化为 LLM 可理解的上下文文本

        Args:
            alert_name: 告警名称

        Returns:
            格式化的上下文文本
        """
        analysis = self.get_alert_analysis(alert_name)
        if "error" in analysis:
            return ""

        parts = []
        parts.append(f"## 知识图谱分析: {analysis['label']}（{analysis['level']}）\n")

        # 根因分析
        if analysis["root_causes"]:
            parts.append("### 可能的根因")
            for i, cause in enumerate(analysis["root_causes"], 1):
                parts.append(f"{i}. **{cause['label']}**（类别: {cause['category']}）")
                if cause["actions"]:
                    for action in cause["actions"]:
                        parts.append(f"   - 处置: {action['label']}（{action['urgency']}）")

        # 级联预测
        if analysis["cascade_chain"]:
            parts.append("\n### ⚠️ 级联风险预测")
            for item in analysis["cascade_chain"]:
                arrow = "→" * item["depth"]
                parts.append(f"- {arrow} **{item['to_label']}**: {item['reason']}")

        # 推荐动作
        if analysis["recommended_actions"]:
            parts.append("\n### 推荐处置动作（按紧急度排序）")
            for action in analysis["recommended_actions"]:
                urgency_label = {
                    "immediate": "🔴 立即",
                    "short_term": "🟡 短期",
                    "long_term": "🟢 长期",
                }
                parts.append(f"- {urgency_label.get(action['urgency'], '')} {action['label']}")

        return "\n".join(parts)

    def _fuzzy_match_alert(self, query: str) -> str | None:
        """模糊匹配告警名称"""
        query_lower = query.lower()

        for keyword, alert in self.ALERT_KEYWORDS.items():
            if keyword in query_lower:
                return alert

        # 精确匹配节点名
        for node in self.graph.nodes:
            node_type: object = self.graph.nodes[node].get("type")
            if node_type == "alert" and query_lower in str(node).lower():
                return str(node)

        return None

    def update_from_triples(self, triples: list) -> dict:
        """从抽取的三元组批量更新图谱

        Args:
            triples: Triple 对象列表（来自 kg_extractor）

        Returns:
            dict: 更新统计 {"nodes_added", "edges_added"}
        """
        nodes_added = 0
        edges_added = 0

        # 类型到颜色的映射
        type_defaults = {
            "alert": {"level": "warning"},
            "root_cause": {"category": "extracted"},
            "action": {"urgency": "short_term"},
            "tool": {"usage": ""},
            "metric": {"unit": ""},
        }

        for t in triples:
            # 添加头实体节点
            if t.head not in self.graph:
                defaults = type_defaults.get(t.head_type, {})
                self.graph.add_node(
                    t.head, type=t.head_type, label=t.head, source="extracted", **defaults
                )
                nodes_added += 1

            # 添加尾实体节点
            if t.tail not in self.graph:
                defaults = type_defaults.get(t.tail_type, {})
                self.graph.add_node(
                    t.tail, type=t.tail_type, label=t.tail, source="extracted", **defaults
                )
                nodes_added += 1

            # 添加关系边（检查是否已存在）
            if not self.graph.has_edge(t.head, t.tail):
                self.graph.add_edge(t.head, t.tail, relation=t.relation, source="extracted")
                edges_added += 1

        logger.info(f"图谱增量更新: +{nodes_added} 节点, +{edges_added} 边")
        return {"nodes_added": nodes_added, "edges_added": edges_added}

    # 学习输入的净化上限：节点名/label 会被序列化进 LLM 检索上下文，
    # 无界文本既拖垮 token 预算也构成持久化注入面（外部反馈可直达此处）
    _MAX_LEARNED_TEXT = 120

    @classmethod
    def _sanitize_learning_text(cls, text: Any) -> str:
        """控制字符剔除 + 截断——学习边界对任意来源的输入兜底"""
        s = str(text or "")
        return "".join(ch for ch in s if ch.isprintable())[: cls._MAX_LEARNED_TEXT]

    def update_from_incident(self, incident_data: dict) -> dict:
        """从已解决的故障事件中学习，增量更新图谱

        Args:
            incident_data: 事件数据，包含 alert_type, root_cause, resolution, cascade_alerts

        Returns:
            dict: 更新统计
        """
        edges_added = 0
        nodes_added = 0

        alert_type = self._sanitize_learning_text(incident_data.get("alert_type", ""))
        root_cause = self._sanitize_learning_text(incident_data.get("root_cause", ""))
        resolution = self._sanitize_learning_text(incident_data.get("resolution", ""))
        cascade_alerts = [
            self._sanitize_learning_text(c) for c in incident_data.get("cascade_alerts", [])
        ]
        incident_id = self._sanitize_learning_text(
            incident_data.get("incident_id", "unknown"),
        )

        # 确保告警节点存在
        if alert_type and alert_type not in self.graph:
            self.graph.add_node(
                alert_type,
                type="alert",
                label=alert_type,
                level="warning",
                source=f"incident_{incident_id}",
            )
            nodes_added += 1

        # 添加根因节点和关系
        if alert_type and root_cause:
            if root_cause not in self.graph:
                self.graph.add_node(
                    root_cause,
                    type="root_cause",
                    label=root_cause,
                    category="incident",
                    source=f"incident_{incident_id}",
                )
                nodes_added += 1
            if not self.graph.has_edge(alert_type, root_cause):
                self.graph.add_edge(
                    alert_type, root_cause, relation="CAUSED_BY", source=f"incident_{incident_id}"
                )
                edges_added += 1

        # 添加处置动作和关系
        if root_cause and resolution:
            if resolution not in self.graph:
                self.graph.add_node(
                    resolution,
                    type="action",
                    label=resolution,
                    urgency="short_term",
                    source=f"incident_{incident_id}",
                )
                nodes_added += 1
            if not self.graph.has_edge(root_cause, resolution):
                self.graph.add_edge(
                    root_cause, resolution, relation="RESOLVED_BY", source=f"incident_{incident_id}"
                )
                edges_added += 1

        # 添加级联关系
        for cascade in cascade_alerts:
            if cascade not in self.graph:
                self.graph.add_node(
                    cascade,
                    type="alert",
                    label=cascade,
                    level="warning",
                    source=f"incident_{incident_id}",
                )
                nodes_added += 1
            if not self.graph.has_edge(alert_type, cascade):
                self.graph.add_edge(
                    alert_type,
                    cascade,
                    relation="MAY_TRIGGER",
                    reason=f"来自事件 {incident_id}",
                    source=f"incident_{incident_id}",
                )
                edges_added += 1

        logger.info(f"从事件 {incident_id} 学习: +{nodes_added} 节点, +{edges_added} 边")
        return {"nodes_added": nodes_added, "edges_added": edges_added}

    def get_graph_stats(self) -> dict:
        """获取图谱统计信息"""
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        relation_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "unknown")
            relation_counts[r] = relation_counts.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
            "relation_types": relation_counts,
        }


# 全局单例
knowledge_graph_service = KnowledgeGraphService()
