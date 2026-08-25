"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等。
集成工程化组件：可观测性、限流、输入安全、审计日志。
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import aiops, alerts, chat, feedback, file, health, kg, memory, multi_diag
from app.config import config
from app.core.circuit_breaker import BREAKER_MILVUS, get_breaker
from app.core.health_registry import health_registry
from app.core.milvus_client import milvus_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 初始化 Prompt 模板管理器
    try:
        from app.core.prompt_manager import prompt_manager

        templates = prompt_manager.list_templates()
        logger.info(f"📋 Prompt 模板加载: {len(templates)} 个")
    except Exception as e:
        logger.warning(f"Prompt 模板加载失败: {e}")

    # 初始化审计日志
    from app.core.audit import audit_logger

    logger.info(f"📝 审计日志: {audit_logger.audit_file}")

    # 初始化工具注册中心
    from app.tools.tool_registry import tool_registry

    logger.info(f"🔧 工具注册: {len(tool_registry.list_tools())} 个")

    # 注册健康探针
    health_registry.register("milvus", _probe_milvus)
    health_registry.register("llm")
    health_registry.register("embedding")
    health_registry.register("rerank")
    health_registry.register("mcp_cls")
    health_registry.register("mcp_monitor")

    # 连接 Milvus（非致命：连接失败不阻止服务启动）
    logger.info("🔌 正在连接 Milvus...")
    try:
        milvus_manager.connect()
        logger.info("✅ Milvus 连接成功")
        health_registry.mark_success("milvus")
    except Exception as e:
        logger.warning(f"⚠️ Milvus 连接失败，服务将以降级模式启动: {e}")
        health_registry.mark_down("milvus")
        get_breaker(BREAKER_MILVUS).record_failure()

    # 长期记忆服务：disabled 时不注册健康组件（避免 /health 常驻 down 的噪音）；
    # enabled 但启动失败才标 down（非致命，降级为无记忆模式继续服务）
    try:
        from app.services.memory import memory_service

        if memory_service.enabled:
            health_registry.register("memory")
            ready = await memory_service.start()
            if not ready:
                health_registry.mark_down("memory")
    except Exception as e:
        logger.warning(f"⚠️ 长期记忆服务启动失败，将以无记忆模式运行: {e}")
        health_registry.mark_down("memory")

    # 定时记忆巩固 worker：随记忆功能启停（周期开关 disabled 或记忆总开关
    # 关闭时不建任务；启动失败非致命——巩固缺失只影响经验沉淀速度）
    try:
        from app.services.memory.consolidation_worker import consolidation_worker

        consolidation_worker.start()
    except Exception as e:
        logger.warning(f"⚠️ 记忆巩固 worker 启动失败: {e}")

    # 启动后台健康探针
    await health_registry.start_probes(interval=config.health_probe_interval)

    logger.info("=" * 60)

    yield

    # 关闭时执行
    await health_registry.stop_probes()
    try:
        from app.services.memory.consolidation_worker import consolidation_worker

        await consolidation_worker.stop()
    except Exception as e:
        logger.warning(f"⚠️ 记忆巩固 worker 停止异常: {e}")
    try:
        from app.services.memory import memory_service

        await memory_service.stop()
    except Exception as e:
        logger.warning(f"⚠️ 长期记忆服务关闭异常: {e}")
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


async def _probe_milvus() -> bool:
    """Milvus 健康探针"""
    return milvus_manager.health_check()


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan,
)

# 配置 CORS（修复旧配置的无效组合：allow_origins=["*"] + credentials=True
# 违反浏览器规范，实际会被浏览器拒绝；通配时强制关闭 credentials）
_origins = list(config.cors_allow_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=config.cors_allow_credentials and "*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册工程化中间件
from app.middleware.auth import APIKeyMiddleware  # noqa: E402
from app.middleware.rate_limiter import RateLimiterMiddleware  # noqa: E402
from app.middleware.request_guard import RequestGuardMiddleware  # noqa: E402

app.add_middleware(RequestGuardMiddleware)
app.add_middleware(RateLimiterMiddleware)
# 鉴权最外层（默认关闭）：无效密钥在限流计数前就被拒绝
app.add_middleware(APIKeyMiddleware)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])
app.include_router(kg.router, prefix="/api", tags=["知识图谱"])
app.include_router(multi_diag.router, prefix="/api", tags=["多Agent诊断"])
app.include_router(feedback.router, prefix="/api", tags=["反馈与评测"])
app.include_router(memory.router, prefix="/api", tags=["长期记忆"])
app.include_router(alerts.router, prefix="/api", tags=["告警接入"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=config.host, port=config.port, reload=config.debug, log_level="info"
    )
