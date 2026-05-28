"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户问题", alias="Question")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？"
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True


class AiRequest(BaseModel):
    """标准化 AI 请求模型

    统一所有入口的请求格式，为鉴权、限流、审计提供基础。
    """
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "anonymous"
    session_id: str = "default"
    scene_code: str = "chat"  # chat / aiops / multi_diagnose / kg_query
    input: str = ""
    variables: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
