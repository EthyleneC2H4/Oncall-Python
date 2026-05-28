"""用户反馈接口 - 诊断结果的用户评价和反馈收集"""

import json
import os
from datetime import datetime

from fastapi import APIRouter
from loguru import logger

from app.models.diagnosis_report import FeedbackRecord
from app.eval.evaluator import AIOpsEvaluator

router = APIRouter()

# 反馈存储文件路径
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "feedback_data.json")


def _load_feedbacks() -> list[dict]:
    """加载所有反馈记录"""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_feedbacks(feedbacks: list[dict]):
    """保存反馈记录"""
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, ensure_ascii=False, indent=2)


@router.post("/feedback")
async def submit_feedback(record: FeedbackRecord):
    """提交用户反馈

    反馈类型：
    - positive: 诊断正确 → 强化该诊断路径置信度
    - negative: 诊断错误 → 记录错误案例
    - comment: 补充信息 → 用户提供实际根因

    Args:
        record: 反馈记录
    """
    logger.info(
        f"收到用户反馈: session={record.session_id}, "
        f"type={record.feedback_type}, comment={record.comment}"
    )

    # 持久化反馈
    feedbacks = _load_feedbacks()
    feedback_entry = record.model_dump()
    feedback_entry["timestamp"] = datetime.now().isoformat()
    feedbacks.append(feedback_entry)
    _save_feedbacks(feedbacks)

    # 如果用户提供了实际根因，学习到知识图谱
    kg_updated = False
    if record.actual_root_cause and record.feedback_type == "negative":
        try:
            from app.services.knowledge_graph_service import knowledge_graph_service
            stats = knowledge_graph_service.update_from_incident({
                "incident_id": f"feedback_{record.session_id}_{record.message_index}",
                "alert_type": record.comment or "未知告警",
                "root_cause": record.actual_root_cause,
                "resolution": "",
                "cascade_alerts": [],
            })
            kg_updated = True
            logger.info(f"从反馈学习到知识图谱: {stats}")
        except Exception as e:
            logger.warning(f"从反馈更新知识图谱失败: {e}")

    return {
        "code": 200,
        "data": {
            "message": "反馈已记录",
            "kg_updated": kg_updated,
            "total_feedbacks": len(feedbacks),
        }
    }


@router.get("/feedback/stats")
async def get_feedback_stats():
    """获取反馈统计"""
    feedbacks = _load_feedbacks()

    stats = {
        "total": len(feedbacks),
        "positive": sum(1 for f in feedbacks if f.get("feedback_type") == "positive"),
        "negative": sum(1 for f in feedbacks if f.get("feedback_type") == "negative"),
        "comments": sum(1 for f in feedbacks if f.get("feedback_type") == "comment"),
    }

    if stats["total"] > 0:
        stats["satisfaction_rate"] = stats["positive"] / stats["total"]
    else:
        stats["satisfaction_rate"] = 0

    return {"code": 200, "data": stats}


@router.post("/eval/run")
async def run_evaluation():
    """运行端到端评测（基础版）

    执行预定义的测试用例，评测路由准确率、检索召回率、KG 根因命中率等。
    优先从 eval/datasets/ 加载 55+ 评测用例。
    """
    logger.info("开始运行 AIOps 评测")
    evaluator = AIOpsEvaluator()
    results = await evaluator.evaluate_all()
    logger.info(f"评测完成: {results.get('summary', {})}")
    return {"code": 200, "data": results}


@router.post("/eval/ragas")
async def run_ragas_evaluation():
    """运行 RAGAS 风格评测（扩展版）

    支持 Context Recall / Context Precision / 路由准确率 等维度。
    使用 eval/datasets/ 下的 55+ 评测用例。
    """
    from app.eval.ragas_evaluator import ragas_evaluator

    logger.info("开始运行 RAGAS 评测")
    results = await ragas_evaluator.evaluate_all()
    logger.info(f"RAGAS 评测完成: {results.get('summary', {})}")
    return {"code": 200, "data": results}


@router.get("/eval/datasets/stats")
async def get_dataset_stats():
    """获取评测数据集统计"""
    from app.eval.ragas_evaluator import ragas_evaluator

    diagnostic = ragas_evaluator.diagnostic_cases
    negative = ragas_evaluator.negative_cases

    # 按类别统计
    categories: dict[str, int] = {}
    for case in diagnostic + negative:
        cat = case.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "code": 200,
        "data": {
            "total": len(diagnostic) + len(negative),
            "diagnostic": len(diagnostic),
            "negative": len(negative),
            "by_category": categories,
        },
    }
