# ─────────────────────────────────────────────────────────────────────────────
# Oncall-Python（SuperBizAgent）生产镜像
#
# 设计要点：
# - uv 锁定安装（uv sync --frozen）：依赖版本由 uv.lock 唯一决定，可复现构建
# - 依赖层与源码层分离：改代码不重装 torch 级重依赖（缓存命中秒级重建）
# - 多阶段：运行时不携带 uv 与构建缓存
# - HuggingFace 模型缓存走卷挂载（HF_HOME），首次启动按需下载 BGE 模型
#
# 构建：docker build -t oncall-python .
# 运行：见 docker-compose.yml（一键部署 + 可选 Milvus/MCP profile）
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: 依赖安装 ────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS deps

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# 先只拷依赖清单：uv.lock 未变时该层稳定命中缓存
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: 运行时 ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# 非 root 运行；数据/日志/模型缓存的命名卷首挂时会继承这里的目录属主
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    # HuggingFace 模型缓存统一收敛到一个可挂卷的路径
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# 运行时资产（.dockerignore 已剔除测试/数据/密钥）
COPY app/ app/
COPY static/ static/
COPY prompts/ prompts/
COPY aiops-docs/ aiops-docs/
COPY mcp_servers/ mcp_servers/
COPY scripts/ scripts/

# 可写目录预先建好并赋属主（命名卷首挂继承）
RUN mkdir -p data logs uploads .cache/huggingface && chown -R appuser:appuser /app
USER appuser

EXPOSE 9900

HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=20 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9900/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
