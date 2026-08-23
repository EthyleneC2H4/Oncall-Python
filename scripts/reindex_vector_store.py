"""Milvus 向量索引重建脚本

Embedding 模型更换（如 DashScope text-embedding-v4 → 本地 BGE）后，
旧向量与新模型的向量空间不兼容，必须重建集合并重灌文档。

用法:
    # 彻底重建：drop 旧集合 → 按新模型重灌 aiops-docs/
    .venv/bin/python scripts/reindex_vector_store.py --drop

    # 仅重灌（保留集合结构，不清空旧数据）
    .venv/bin/python scripts/reindex_vector_store.py

    # 指定文档目录与 sanity 检查查询
    .venv/bin/python scripts/reindex_vector_store.py --drop \
        --docs aiops-docs --sanity-query "CPU 使用率过高如何排查"

前置条件:
    - Milvus 已启动 (make up)
    - .env 中已配置 OPENROUTER_API_KEY（sanity 检索不需要，但服务启动需要）
    - 本地 BGE 模型首次运行会自动下载 (~1.3GB)
"""

import argparse
import sys
from pathlib import Path

# 保证以 `python scripts/xxx.py` 运行时也能导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建 Milvus 向量索引")
    parser.add_argument(
        "--docs",
        default="aiops-docs",
        help="待索引的文档目录 (默认: aiops-docs)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="先删除旧 collection 再重建（更换 embedding 模型时必须）",
    )
    parser.add_argument(
        "--sanity-query",
        default="CPU 使用率过高如何排查",
        help="重建完成后的检索冒烟查询",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="跳过检索冒烟检查",
    )
    return parser.parse_args()


def drop_collection_if_exists() -> None:
    """删除现有 collection（连接级别操作，不触发自动建表）"""
    from pymilvus import connections, utility

    from app.config import config

    logger.warning("--drop 已指定：将删除旧 collection 并重建")
    connections.connect(
        alias="reindex",
        host=config.milvus_host,
        port=str(config.milvus_port),
        timeout=config.milvus_timeout / 1000,
    )
    if utility.has_collection("biz", using="reindex"):
        utility.drop_collection("biz", using="reindex")
        logger.info("旧 collection 'biz' 已删除")
    else:
        logger.info("collection 'biz' 不存在，无需删除")
    connections.disconnect("reindex")


def main() -> int:
    args = parse_args()

    docs_dir = Path(args.docs).resolve()
    if not docs_dir.is_dir():
        logger.error(f"文档目录不存在: {docs_dir}")
        return 1

    # Step 1: 可选 drop（必须在任何 app 侧 Milvus 连接之前执行）
    if args.drop:
        drop_collection_if_exists()

    # Step 2: 连接 Milvus（集合不存在会按当前配置维度自动创建）
    from app.core.milvus_client import milvus_manager

    milvus_manager.connect()
    logger.info(f"Milvus 就绪, 向量维度: {milvus_manager.VECTOR_DIM}")

    # Step 3: 重灌文档（走既有分块 + 本地 BGE 向量化链路）
    from app.services.vector_index_service import vector_index_service

    result = vector_index_service.index_directory(str(docs_dir))
    summary = result.to_dict()
    logger.info(f"索引结果: {summary}")

    if not result.success:
        logger.error(
            f"部分文件索引失败 ({result.fail_count}/{result.total_files})，"
            f"失败清单: {result.failed_files}"
        )
        return 2

    # Step 4: 检索冒烟验证
    if not args.skip_sanity:
        from app.services.vector_store_manager import vector_store_manager

        docs = vector_store_manager.similarity_search(args.sanity_query, k=3)
        if not docs:
            logger.error(f"检索冒烟失败: '{args.sanity_query}' 无任何返回")
            return 3
        logger.info(f"检索冒烟通过: 命中 {len(docs)} 条")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("_file_name", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            logger.info(f"  [{i}] {source} | {preview}...")

    logger.info("✅ 向量索引重建完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
