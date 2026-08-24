"""长期记忆 - 类型定义

四类记忆模型（参照认知科学分层，借鉴 Cortex memory 模块）：
- working:   当前会话的即时状态（不落库，由 LangGraph checkpointer 承担）
- episodic:  情景记忆 —— 一次具体交互/诊断事件的原样记录（"上次发生了什么"）
- semantic:  语义记忆 —— 从情景中巩固出的通用知识/事故经验（"我们知道什么"）
- procedural:程序记忆 —— 可复用的操作步骤/处置方案（"该怎么做"）

打分权重与新近度衰减参数从 app.config 读取；本模块保持纯数据定义，
不依赖 sqlite / embedding 等基础设施，便于单测。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    """记忆类型（四层）"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryItem:
    """单条长期记忆

    Attributes:
        id: 全局唯一 ID（uuid4 hex）
        type: 记忆类型（四层之一）
        content: 自然语言内容（检索的主体）
        importance: 重要性 0.0~1.0（写入时评估；召回打分的静态分量）
        embedding: 内容向量（BGE 归一化向量）；None 表示尚未嵌入
        user_id: 归属用户/命名空间（作品集场景默认单用户 "local"）
        session_id: 产生该记忆的会话来源（可空）
        metadata: 任意扩展信息（工具调用统计、来源标记等）
        created_at: 创建时间（epoch 秒）
        last_accessed_at: 最近一次被召回的时间（epoch 秒）
        access_count: 被召回次数
        deleted_at: 软删除时间（epoch 秒）；None 表示未删除
        consolidated_into: 巩固后指向的语义记忆 id（情景被合并时回填）
    """

    type: MemoryType
    content: str
    importance: float = 0.3
    embedding: list[float] | None = None
    user_id: str = "local"
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    deleted_at: float | None = None
    consolidated_into: str | None = None

    def touch(self, at: float | None = None) -> None:
        """记录一次召回访问"""
        now = time.time() if at is None else at
        self.last_accessed_at = now
        self.access_count += 1

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """序列化为 API/日志友好的 dict（向量默认不输出）"""
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "importance": round(self.importance, 4),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "deleted": self.deleted_at is not None,
            "consolidated_into": self.consolidated_into,
        }
        if include_embedding:
            data["embedding_dim"] = len(self.embedding) if self.embedding else 0
        return data


def new_memory_id() -> str:
    """生成新的记忆 ID"""
    return uuid.uuid4().hex
