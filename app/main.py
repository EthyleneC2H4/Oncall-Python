"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops, kg, multi_diag, feedback
from app.core.milvus_client import milvus_manager
from app.core.health_registry import health_registry
from app.core.circuit_breaker import get_breaker, BREAKER_MILVUS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 注册健康探针
    health_registry.register("milvus", _probe_milvus)
    health_registry.register("dashscope_llm")
    health_registry.register("dashscope_embedding")
    health_registry.register("dashscope_rerank")
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

    # 启动后台健康探针
    await health_registry.start_probes(interval=config.health_probe_interval)

    logger.info("=" * 60)

    yield

    # 关闭时执行
    await health_registry.stop_probes()
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
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])
app.include_router(kg.router, prefix="/api", tags=["知识图谱"])
app.include_router(multi_diag.router, prefix="/api", tags=["多Agent诊断"])
app.include_router(feedback.router, prefix="/api", tags=["反馈与评测"])

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
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
