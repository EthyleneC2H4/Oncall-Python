"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import json
import re

from fastapi import APIRouter, Header, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()

# P5 AB：变体名白名单（请求头 X-Prompt-Variant），防注入与超长
_VARIANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _sanitize_prompt_variant(raw: str | None) -> str | None:
    """清洗 X-Prompt-Variant 请求头；不合法值静默按基线处理（不报错打断对话）

    变体名约定全小写（YAML variants 键与 cost_tracker 分组均按小写登记），
    这里统一归一小写，避免 "Concise" 与 "concise" 被当成两个分组。
    """
    if not raw:
        return None
    value = raw.strip().lower()
    return value if _VARIANT_RE.match(value) else None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    x_prompt_variant: str | None = Header(default=None, alias="X-Prompt-Variant"),
):
    """快速对话接口
    {
        "code": 200,
        "message": "success",
        "data": {
            "success": true,
            "answer": "回答内容",
            "errorMessage": null
        }
    }

    Args:
        request: 对话请求
        x_prompt_variant: Prompt AB 变体名（可选请求头）

    Returns:
        统一格式的对话响应
    """
    try:
        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}")
        answer = await rag_agent_service.query(
            request.question,
            session_id=request.id,
            prompt_variant=_sanitize_prompt_variant(x_prompt_variant),
        )

        logger.info(f"[会话 {request.id}] 快速对话完成")

        return {
            "code": 200,
            "message": "success",
            "data": {"success": True, "answer": answer, "errorMessage": None},
        }

    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {"success": False, "answer": None, "errorMessage": str(e)},
        }


@router.post("/chat_stream")
async def chat_stream(
    request: ChatRequest,
    x_prompt_variant: str | None = Header(default=None, alias="X-Prompt-Variant"),
):
    """流式对话接口（基于 RAG Agent，SSE）

    返回 SSE 格式，data 字段为 JSON：

    工具调用事件:
    event: message
    data: {"type":"tool_call","data":{"tool":"工具名","status":"start|end","input":{...}}}

    内容流式事件:
    event: message
    data: {"type":"content","data":"内容块"}

    完成事件:
    event: message
    data: {"type":"done","data":{"answer":"完整答案","tool_calls":[...]}}
          （P5：变体实际生效时 done.data 为 {"prompt_variant": "生效变体"}；
          基线运行（含未登记/渲染失败回退）保持原形状不变）

    Args:
        request: 对话请求
        x_prompt_variant: Prompt AB 变体名（可选请求头）

    Returns:
        SSE 事件流
    """
    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}")
    prompt_variant = _sanitize_prompt_variant(x_prompt_variant)

    async def event_generator():
        try:
            async for chunk in rag_agent_service.query_stream(
                request.question, session_id=request.id, prompt_variant=prompt_variant
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                # 处理调试类型消息（新增）
                if chunk_type == "debug":
                    # 调试信息，可以选择发送或忽略
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "debug",
                                "node": chunk.get("node", "unknown"),
                                "message_type": chunk.get("message_type", "unknown"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "tool_call":
                    # 发送工具调用事件（可选，前端可以显示工具调用状态）
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "tool_call", "data": chunk_data}, ensure_ascii=False
                        ),
                    }
                elif chunk_type == "search_results":
                    # 发送检索结果（可选，前端可以忽略）
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "search_results", "data": chunk_data}, ensure_ascii=False
                        ),
                    }
                elif chunk_type == "content":
                    # 发送内容块 - 关键：data 必须是 JSON 字符串
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "content", "data": chunk_data}, ensure_ascii=False
                        ),
                    }
                elif chunk_type == "complete":
                    # 发送完成信号；P5：变体运行时附带生效变体名（只增不改）
                    done_data = (
                        {"prompt_variant": chunk["prompt_variant"]}
                        if chunk.get("prompt_variant")
                        else chunk_data
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "done", "data": done_data}, ensure_ascii=False
                        ),
                    }
                elif chunk_type == "error":
                    # 发送错误信息
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "error", "data": str(chunk_data)}, ensure_ascii=False
                        ),
                    }

            logger.info(f"[会话 {request.id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None,
        )

    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """查询会话历史

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    try:
        history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id, message_count=len(history), history=history
        )

    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
