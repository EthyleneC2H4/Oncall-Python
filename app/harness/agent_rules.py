"""运维诊断规则 - 基于历史失败案例积累的经验规则

每条规则对应一个历史失败模式，帮助 Agent 避免常见的诊断错误。
这些规则会被注入到 Executor 的系统提示词中，作为 Harness 层的约束。
"""

AGENT_RULES = """
## 运维诊断规则（每条对应一个历史失败案例）

### 关联分析规则
1. CPU 高告警分析时，必须同时检查内存使用率，因为 OOM 导致的 Full GC 会表现为 CPU 飙高
2. 磁盘告警排查时，优先检查日志目录和临时文件，这是最常见的磁盘满原因
3. 网络延迟问题不要只看单个服务，需要检查上下游依赖链路
4. 响应变慢时要区分"慢查询导致"和"流量突增导致"，两者处置方案完全不同

### 处置建议规则
5. 给出处置建议时，必须区分"紧急操作"（立即执行）和"长期优化"（计划执行）
6. 重启操作建议前，必须先建议保存现场（dump 内存、保留日志），否则丢失诊断线索
7. 扩容建议必须同时评估成本影响，不要无条件推荐扩容

### 诊断纪律规则
8. 不要在没有证据的情况下推测根因，如果信息不足，建议用户提供更多数据
9. 多个告警同时出现时，优先分析可能是根因的告警（通常是最早触发的），避免被级联告警误导
10. 工具调用失败时，不要忽略失败结果，必须在报告中说明哪些数据未能获取
"""

# 按告警类型索引的特定规则
_ALERT_SPECIFIC_RULES = {
    "cpu": [
        "CPU 告警必须同时查看内存和 GC 状态，排除 OOM 引发的 Full GC",
        "检查是否有定时任务重叠执行（cron overlap）",
    ],
    "memory": [
        "内存告警先区分是泄漏还是配置不当（JVM heap 太小 vs 代码泄漏）",
        "建议导出 heap dump 再重启，保留诊断线索",
    ],
    "disk": [
        "磁盘告警优先检查 /var/log 和 /tmp 目录",
        "检查是否配置了日志轮转（logrotate）",
    ],
    "slow": [
        "响应慢先看 P99 而非平均值，平均值会掩盖尾部延迟",
        "区分上游慢（外部 API 超时）和本地慢（CPU/DB 慢查询）",
    ],
    "unavailable": [
        "服务不可用先检查健康检查端点是否正常",
        "检查依赖服务状态（数据库、缓存、消息队列）",
    ],
}


def get_rules_for_alert(alert_keyword: str) -> str:
    """获取特定告警类型的规则

    Args:
        alert_keyword: 告警关键词

    Returns:
        str: 相关规则文本
    """
    rules = [AGENT_RULES]

    keyword_lower = alert_keyword.lower()
    for key, specific_rules in _ALERT_SPECIFIC_RULES.items():
        if key in keyword_lower:
            rules.append(f"\n### {alert_keyword} 专项规则")
            for _i, rule in enumerate(specific_rules, 1):
                rules.append(f"- {rule}")

    return "\n".join(rules)
