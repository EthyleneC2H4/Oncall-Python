"""健康检查接口"""

from typing import Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import config
from app.core.milvus_client import milvus_manager
from app.core.health_registry import health_registry
from app.core.cache import embedding_cache, llm_response_cache, retrieval_cache
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查接口

    返回所有服务的健康状态和缓存统计。

    Returns:
        JSONResponse: 健康检查结果
    """
    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy",
    }

    # 从健康注册中心获取所有服务状态
    services_status = health_registry.get_all_status()
    health_data["services"] = services_status

    # 兼容旧逻辑：单独检查 Milvus
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    # 缓存统计
    health_data["caches"] = {
        "embedding": embedding_cache.stats,
        "llm_response": llm_response_cache.stats,
        "retrieval": retrieval_cache.stats,
    }

    # 判断整体健康状态
    overall_status = "healthy"
    status_code = 200

    # 检查是否有 DOWN 的服务
    down_services = [
        name for name, info in services_status.items()
        if info.get("status") == "down"
    ]

    if down_services:
        # 仅 Milvus DOWN 时标记 degraded（非 unhealthy），因为 BM25 可兜底
        if down_services == ["milvus"]:
            overall_status = "degraded"
            status_code = 200
            health_data["degraded_services"] = down_services
        elif "dashscope_llm" in down_services:
            overall_status = "unhealthy"
            status_code = 503
            health_data["error"] = f"关键服务不可用: {', '.join(down_services)}"
        else:
            overall_status = "degraded"
            status_code = 200
            health_data["degraded_services"] = down_services

    health_data["status"] = overall_status

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": {
                "healthy": "服务运行正常",
                "degraded": "服务降级运行中",
                "unhealthy": "服务不可用",
            }.get(overall_status, "未知状态"),
            "data": health_data
        }
    )
