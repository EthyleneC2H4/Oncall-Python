"""长期记忆接口 - 用户记忆的查询、统计与清除"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.services.memory import MemoryType, memory_service

router = APIRouter()


@router.get("/memory/{user_id}")
async def get_user_memory(
    user_id: str,
    type: str | None = None,  # noqa: A002 - 对外参数名沿用业务语义
    limit: int = 100,
    include_deleted: bool = False,
):
    """获取指定用户的长期记忆列表

    Args:
        user_id: 用户 ID（记忆命名空间）
        type: 可选类型过滤（working/episodic/semantic/procedural，逗号分隔）
        limit: 返回上限（默认 100）
        include_deleted: 是否包含已软删除条目
    """
    try:
        types: list[MemoryType] | None = None
        if type:
            types = [MemoryType(t.strip()) for t in type.split(",") if t.strip()]
        items = await memory_service.list_items(
            user_id=user_id,
            types=types,
            include_deleted=include_deleted,
            limit=limit,
        )
        stats = await memory_service.stats()
        logger.info(f"Memory API: 查询用户 {user_id} 记忆，返回 {len(items)} 条")
        return {
            "code": 200,
            "data": {
                "user_id": user_id,
                "total": len(items),
                "memories": [item.to_dict() for item in items],
                "stats": stats,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效的记忆类型: {e}") from e
    except Exception as e:
        logger.error(f"查询用户记忆失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询用户记忆失败: {e}") from e


@router.delete("/memory/{user_id}")
async def delete_user_memory(user_id: str):
    """清除指定用户的全部长期记忆（软删除）

    Args:
        user_id: 用户 ID（记忆命名空间）
    """
    try:
        deleted = await memory_service.forget_user(user_id)
        logger.info(f"Memory API: 清除用户 {user_id} 记忆 {deleted} 条")
        return {"code": 200, "data": {"user_id": user_id, "deleted": deleted}}
    except Exception as e:
        logger.error(f"清除用户记忆失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除用户记忆失败: {e}") from e


@router.post("/memory/consolidate")
async def trigger_memory_consolidation():
    """手动触发一次情景→语义记忆巩固（与周期 worker 共用同一入口）

    Returns:
        巩固统计：{"clusters", "members_consolidated", "semantic_ids"}
    """
    from app.services.memory.consolidation_worker import consolidation_worker

    stats = await consolidation_worker.run_once()
    if stats is None:  # 记忆服务未就绪（总开关关闭/建库失败）
        raise HTTPException(status_code=409, detail="长期记忆未启用，无法执行巩固")
    logger.info(f"Memory API: 手动触发记忆巩固 {stats}")
    return {"code": 200, "data": stats}
