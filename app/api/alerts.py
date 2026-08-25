"""告警 webhook 接入通道（C2）

接收 Alertmanager 兼容的 webhook payload（容忍单告警对象与缺省字段）：

- firing   → 组装诊断任务文本，经 BackgroundTasks 在响应返回后触发一次
             AIOps 诊断（告警管道不被分钟级的诊断流程阻塞）；
- resolved → 尽力沉淀进知识图谱（update_from_incident；仅当注解显式携带
             root_cause/resolution 时才学习——描述文本不是根因，不污染图谱）。

安全与卫生：
- 可选共享密钥：配置 alert_webhook_token 后强制校验请求头 X-Alert-Token；
- 指纹去重窗口：Alertmanager 重发/grouped 重试在窗口内只触发一次诊断。
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from loguru import logger

from app.agent.runtime.events import EventType
from app.config import config
from app.services.aiops_service import aiops_service

router = APIRouter()

# 进程内指纹去重：key → 上次接受时间。检查与写入之间无 await，
# 事件循环单线程语义下无需加锁；每次请求顺手清理过期项防膨胀。
_dedup: dict[str, float] = {}


def _extract_alerts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Alertmanager 结构（alerts 数组）优先；兼容单告警对象直投"""
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        return [a for a in alerts if isinstance(a, dict)]
    if "alertname" in payload or "labels" in payload:
        return [payload]
    return []


def _normalize(raw: dict[str, Any], group_status: str) -> dict[str, Any] | None:
    """把一条原始告警归一化；无 alertname 的条目丢弃"""
    labels = raw.get("labels") or {}
    annotations = raw.get("annotations") or {}
    name = str(labels.get("alertname") or raw.get("alertname") or "").strip()
    if not name:
        return None
    status = str(raw.get("status") or group_status or "firing").lower()
    return {
        "alertname": name,
        "status": status,
        "severity": str(labels.get("severity", "warning")),
        "instance": str(labels.get("instance", "")),
        "service": str(labels.get("service") or labels.get("job") or ""),
        "description": str(annotations.get("description") or annotations.get("summary") or ""),
        "root_cause": str(annotations.get("root_cause") or ""),
        "resolution": str(annotations.get("resolution") or ""),
        "starts_at": str(raw.get("startsAt") or raw.get("starts_at") or ""),
        "fingerprint": str(raw.get("fingerprint") or ""),
    }


def build_diagnosis_task(alerts: list[dict[str, Any]]) -> str:
    """把归一化告警组装成诊断任务提示词"""
    lines = ["收到以下告警（webhook 通道），请针对这些告警进行根因分析并生成诊断报告：", ""]
    for i, a in enumerate(alerts, 1):
        line = f"{i}. [{a['severity']}] {a['alertname']}"
        if a["instance"]:
            line += f" @ {a['instance']}"
        lines.append(line)
        if a["service"]:
            lines.append(f"   服务: {a['service']}")
        if a["description"]:
            lines.append(f"   描述: {a['description']}")
        if a["starts_at"]:
            lines.append(f"   开始时间: {a['starts_at']}")
    lines += ["", "请查询相关指标、日志与知识库，给出根因分析与处置建议。"]
    return "\n".join(lines)


async def run_webhook_diagnosis(task_text: str, session_id: str) -> None:
    """后台执行一次告警诊断：消费事件流到终止事件为止（仅日志观测）"""
    logger.info(f"[webhook {session_id}] 后台诊断启动")
    try:
        async for ev in aiops_service.execute(task_text, session_id=session_id):
            if ev.type in (EventType.COMPLETE, EventType.ERROR):
                break
        logger.info(f"[webhook {session_id}] 后台诊断结束")
    except Exception as e:
        logger.error(f"[webhook {session_id}] 后台诊断异常: {e}", exc_info=True)


def _learn_resolved(alert: dict[str, Any], sibling_names: list[str]) -> bool:
    """resolved 告警尽力学习进知识图谱；无显式根因/处置时不学（防污染）"""
    if not alert["root_cause"] and not alert["resolution"]:
        return False
    try:
        from app.services.knowledge_graph_service import knowledge_graph_service

        stats = knowledge_graph_service.update_from_incident(
            {
                "incident_id": f"webhook_{alert['fingerprint'] or uuid.uuid4().hex[:8]}",
                "alert_type": alert["alertname"],
                "root_cause": alert["root_cause"],
                "resolution": alert["resolution"],
                "cascade_alerts": [n for n in sibling_names if n != alert["alertname"]],
            }
        )
        logger.info(f"从 resolved 告警学习到知识图谱: {stats}")
        return True
    except Exception as e:  # noqa: BLE001 - 学习失败不影响 webhook 主流程
        logger.warning(f"从 resolved 告警学习知识图谱失败: {e}")
        return False


@router.post("/alerts/webhook")
async def alerts_webhook(request: Request, background_tasks: BackgroundTasks):
    """接收 Alertmanager 兼容告警：firing 触发后台诊断，resolved 尽力学习图谱"""
    # 共享密钥（可选）：留空则不校验（本地开发零负担）
    expected = config.alert_webhook_token
    if expected and request.headers.get("x-alert-token", "") != expected:
        raise HTTPException(status_code=401, detail="X-Alert-Token 校验失败")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload 不是合法 JSON: {e}") from e
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload 必须是 JSON 对象")

    group_status = str(payload.get("status") or "firing")
    raw_alerts = _extract_alerts(payload)
    if not raw_alerts:
        raise HTTPException(
            status_code=422,
            detail="未找到可处理的告警条目（需要 alerts 数组或单告警对象字段）",
        )

    now = time.time()
    window = config.alert_webhook_dedup_window_seconds
    for key, ts in list(_dedup.items()):  # 清理过期项，dict 不随时间无界膨胀
        if now - ts >= window:
            _dedup.pop(key, None)

    accepted: list[dict[str, Any]] = []
    suppressed = 0
    learned: list[str] = []
    all_names = [(a or {}).get("labels", {}).get("alertname", "") for a in raw_alerts]
    for raw in raw_alerts:
        alert = _normalize(raw, group_status)
        if alert is None:
            continue
        if alert["status"] == "resolved":
            if _learn_resolved(alert, all_names):
                learned.append(alert["alertname"])
            continue
        key = alert["fingerprint"] or f"{alert['alertname']}@{alert['instance']}"
        if key in _dedup and now - _dedup[key] < window:
            suppressed += 1  # 窗口内重复告警：不重复触发诊断
            continue
        _dedup[key] = now
        accepted.append(alert)

    session_id = ""
    task_preview = ""
    if accepted:
        session_id = f"webhook-{uuid.uuid4().hex[:12]}"
        task_text = build_diagnosis_task(accepted)
        task_preview = task_text[:300]
        background_tasks.add_task(run_webhook_diagnosis, task_text, session_id)
        logger.info(
            f"[webhook {session_id}] 接受 {len(accepted)} 条 firing 告警"
            f"（抑制 {suppressed} 条重复），后台诊断已排队"
        )

    return {
        "code": 202,
        "data": {
            "accepted": len(accepted),
            "suppressed": suppressed,
            "resolved_learned": learned,
            "session_id": session_id,
            "task_preview": task_preview,
        },
    }
