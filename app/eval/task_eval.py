"""GAIA 式任务级评测 —— 按必需证据子串分级（纯函数，离线可跑）

借鉴 GAIA 的分级思想但适配运维诊断场景：
不要求答案逐字等于金标，而是检查「必需证据」是否出现在最终报告里：

- EXACT   (1.0)：所有必需证据命中，且无禁词
- PARTIAL (0.5)：命中至少一条必需证据（或配置的最低命中率）
- WRONG   (0.0)：必需证据全部缺失，或命中禁词

配套 dataset_registry 版本化的用例格式：
    {"id": "TC001", "query": "...",
     "required_evidence": ["内存泄漏", "DumpMemory"],
     "forbidden_evidence": [], "weight": 1.0}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskGrade(StrEnum):
    """任务完成度等级"""

    EXACT = "exact"
    PARTIAL = "partial"
    WRONG = "wrong"


GRADE_SCORES: dict[TaskGrade, float] = {
    TaskGrade.EXACT: 1.0,
    TaskGrade.PARTIAL: 0.5,
    TaskGrade.WRONG: 0.0,
}


@dataclass
class TaskVerdict:
    """单个任务的评测结论（score 为未加权等级分，weight 独立携带）"""

    case_id: str
    grade: TaskGrade
    score: float
    weight: float = 1.0
    hit_required: list[str] = field(default_factory=list)
    missed_required: list[str] = field(default_factory=list)
    hit_forbidden: list[str] = field(default_factory=list)


def _contains(haystack: str, needle: str) -> bool:
    """大小写不敏感的子串判断"""
    return needle.lower() in haystack.lower()


def grade_answer(
    answer: str,
    required_evidence: list[str],
    forbidden_evidence: list[str] | None = None,
    *,
    partial_threshold: float = 0.5,
) -> TaskVerdict:
    """按必需/禁止证据为一次回答定级（纯函数，weight 恒为默认值）

    Args:
        answer: 系统生成的最终回答
        required_evidence: 必须出现的证据子串（全部命中 → exact）
        forbidden_evidence: 出现即判 wrong 的子串（如错误处置建议）
        partial_threshold: 命中率达到该比例判 partial，否则 wrong

    契约：required 全空时视为「无法判定」——直接 exact（空期望不惩罚）。
    """
    text = answer or ""
    forbidden = forbidden_evidence or []

    hit_forbidden = [e for e in forbidden if e and _contains(text, e)]
    if hit_forbidden:
        return TaskVerdict(
            case_id="", grade=TaskGrade.WRONG, score=GRADE_SCORES[TaskGrade.WRONG],
            hit_forbidden=hit_forbidden,
        )

    effective = [e for e in required_evidence if e]
    if not effective:
        return TaskVerdict(case_id="", grade=TaskGrade.EXACT, score=1.0)

    hit = [e for e in effective if _contains(text, e)]
    missed = [e for e in effective if not _contains(text, e)]

    hit_ratio = len(hit) / len(effective)
    if not missed:
        grade = TaskGrade.EXACT
    elif hit_ratio >= partial_threshold:
        grade = TaskGrade.PARTIAL
    else:
        grade = TaskGrade.WRONG

    return TaskVerdict(
        case_id="", grade=grade, score=GRADE_SCORES[grade],
        hit_required=hit, missed_required=missed,
    )


def evaluate_case(case: dict, answer: str) -> TaskVerdict:
    """按数据集用例 schema 评测定向单例（补全 case_id 与 weight 元信息）"""
    verdict = grade_answer(
        answer,
        list(case.get("required_evidence") or []),
        list(case.get("forbidden_evidence") or []),
        partial_threshold=float(case.get("partial_threshold", 0.5)),
    )
    verdict.case_id = str(case.get("id", ""))
    verdict.weight = float(case.get("weight", 1.0))
    return verdict


def summarize(verdicts: list[TaskVerdict]) -> dict:
    """聚合一批结论：权重归一平均分 + 分级分布"""
    distribution: dict[str, int] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for v in verdicts:
        distribution[v.grade.value] = distribution.get(v.grade.value, 0) + 1
        total_weight += v.weight
        weighted_sum += v.score * v.weight

    return {
        "total": len(verdicts),
        "avg_score": (
            round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
        ),
        "grades": distribution,
    }
