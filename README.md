# SuperBizAgent

> 企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能诊断

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)

## ✨ 核心特性

- 🤖 **智能对话** — LangChain 多轮对话 + 流式输出 + Token 预算上下文引擎（类型化 Packet + 分配额装配 + LLM roll-up 压缩）
- 🧵 **长期记忆** — 情景/语义/程序三类长期记忆（sqlite 持久化），向量召回 + 加权打分，情景→语义巩固的经验沉淀闭环
- 📚 **全链路 RAG** — Query Rewrite → 向量 + BM25 双路召回 → RRF 融合 → Rerank 精排 + 结构感知 Parent-Child 分块
- 🔧 **AIOps 诊断** — Plan-Execute-Replan 自主诊断工作流 + Harness 诊断规则约束
- 🧠 **知识图谱** — 运维领域知识图谱，支持告警根因分析、级联预测、LLM 自动三元组抽取
- 👥 **多 Agent 并行** — Coordinator-Specialist-Synthesizer 架构，日志/指标/知识三维度并行诊断
- 🎯 **查询意图路由** — 4 类意图智能分类，自动匹配最优检索策略
- 📊 **评测与反馈** — 端到端评测框架 + 用户反馈闭环 + 知识图谱自进化
- 🔌 **MCP 集成** — 标准化工具协议接入日志查询和监控指标

## 🛠️ 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: OpenRouter（OpenAI 兼容端点，默认 NVIDIA Nemotron 3.5 Lightning）
- **向量库**: Milvus (IVF_FLAT / L2)
- **Embedding**: 本地 BGE (BAAI/bge-large-zh-v1.5, sentence-transformers)
- **稀疏检索**: BM25 (rank_bm25 + jieba)
- **重排**: 本地 BGE Reranker (BAAI/bge-reranker-base, Cross-Encoder)
- **知识图谱**: NetworkX (DiGraph)
- **工具协议**: MCP (Model Context Protocol)
- **前端**: 原生 HTML/CSS/JS + vis-network (图谱可视化)

## 🚀 快速开始

### 环境要求
- Python 3.10+
- [OpenRouter](https://openrouter.ai/) API Key ([获取地址](https://openrouter.ai/settings/keys))

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 安装依赖（推荐使用 uv）
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 方式 2: 使用 pip
pip install -e .

# 3. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 OPENROUTER_API_KEY
vim .env  # 或使用其他编辑器

# 4. 一键初始化（启动 Docker + 服务 + 上传文档）
make init

# 5. 一键启动
make start
```

#### Windows 环境（PowerShell/CMD）

如果Windows 不支持 `make` 命令，可以手动执行以下步骤以启动服务：

```powershell
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
# 创建虚拟环境
uv venv
# 激活虚拟环境
.venv\Scripts\activate
# 安装所有依赖
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 OPENROUTER_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 等待 Milvus 启动完成（约 5-10 秒）
timeout /t 10

# 7. 启动 MCP 服务
# 启动 CLS 日志查询服务（新开一个 PowerShell 窗口）
python mcp_servers/cls_server.py

# 启动 Monitor 监控服务（新开一个 PowerShell 窗口）
python mcp_servers/monitor_server.py

# 8. 启动 FastAPI 主服务（新开一个 PowerShell 窗口）
# 注意：日志会自动输出到 logs\app_YYYY-MM-DD.log
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 9. 上传文档到向量库（新开一个 PowerShell 窗口）
# 等待服务启动完成后执行
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）

使用启动脚本：

```powershell
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| AIOps 诊断 | POST | `/api/aiops` | Plan-Execute-Replan 自动诊断 (SSE) |
| 多 Agent 诊断 | POST | `/api/multi-diagnose` | 三 Agent 并行诊断 (SSE) |
| KG 根因分析 | GET | `/api/kg/analyze/{keyword}` | 告警根因 + 处置方案 |
| KG 级联预测 | GET | `/api/kg/cascade/{keyword}` | BFS 级联风险预测 |
| KG 图谱数据 | GET | `/api/kg/graph` | 完整图谱 (vis-network) |
| KG 三元组抽取 | POST | `/api/kg/extract` | 从文档自动抽取三元组 |
| KG 事件学习 | POST | `/api/kg/learn-incident` | 从故障事件增量学习 |
| 用户反馈 | POST | `/api/feedback` | 提交反馈 (负反馈自动学习到 KG) |
| 记忆查询 | GET | `/api/memory/{user_id}` | 长期记忆列举（type 过滤 + 统计） |
| 记忆遗忘 | DELETE | `/api/memory/{user_id}` | 软删除某用户全部长期记忆 |
| 评测运行 | POST | `/api/eval/run` | 端到端评测 (5 场景 × 4 维度) |
| 文件上传 | POST | `/api/upload` | 上传并索引文档 |
| 健康检查 | GET | `/health` | 服务状态检查 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer
```

## 📁 项目结构

```
Oncall-Python/
├── app/                                    # 核心应用
│   ├── main.py                             # FastAPI 入口 + 生命周期管理
│   ├── config.py                           # Pydantic Settings 配置管理
│   ├── api/                                # REST API 路由层
│   │   ├── chat.py                         #   对话接口（快速/流式）
│   │   ├── aiops.py                        #   AIOps 诊断接口 (SSE)
│   │   ├── kg.py                           #   知识图谱接口（分析/级联/抽取/学习）
│   │   ├── multi_diag.py                   #   多 Agent 并行诊断接口
│   │   ├── feedback.py                     #   反馈收集 + 评测运行接口
│   │   ├── memory.py                       #   长期记忆查询/遗忘接口
│   │   ├── event_translator.py             #   运行时事件 → 旧版 SSE dict 翻译器
│   │   ├── file.py                         #   文件上传 + 知识库索引接口
│   │   └── health.py                       #   健康检查接口
│   ├── services/                           # 业务逻辑层
│   │   ├── rag_agent_service.py            #   RAG Agent（LangGraph + 五层结构化提示词）
│   │   ├── aiops_service.py                #   AIOps 工作流编排（Plan-Execute-Replan）
│   │   ├── knowledge_graph_service.py      #   运维知识图谱（NetworkX）
│   │   ├── kg_extractor.py                 #   LLM 自动三元组抽取
│   │   ├── query_router.py                 #   查询意图分类与路由（4 类意图）
│   │   ├── query_rewriter.py               #   LLM 查询改写（口语化→规范化）
│   │   ├── bm25_retriever.py               #   BM25 稀疏检索（jieba + rank_bm25）
│   │   ├── reranker.py                     #   Rerank 精排（本地 bge-reranker）
│   │   ├── context_assembler.py            #   动态上下文组装（Token 优先级）
│   │   ├── memory/                         #   长期记忆服务
│   │   │   ├── types.py                    #     MemoryItem + 四类记忆类型
│   │   │   ├── store.py                    #     sqlite 存储层（WAL + 软删除）
│   │   │   ├── scoring.py                  #     打分纯函数（相关性/重要性/时近衰减）
│   │   │   ├── queue.py                    #     单 worker 异步写队列
│   │   │   └── service.py                  #     服务门面（write/recall/consolidate/forget）
│   │   ├── document_splitter_service.py    #   结构感知分块 + Parent-Child 双层索引
│   │   ├── vector_store_manager.py         #   Milvus 向量存储管理
│   │   ├── vector_embedding_service.py     #   本地 BGE 嵌入服务
│   │   ├── vector_index_service.py         #   文档索引服务
│   │   └── vector_search_service.py        #   语义搜索服务
│   ├── agent/                              # Agent 模块
│   │   ├── mcp_client.py                   #   MCP 客户端（带重试拦截器）
│   │   ├── runtime/                        #   统一 Agent 运行时
│   │   │   ├── base.py                     #     AgentRuntime ABC + RuntimeRegistry
│   │   │   ├── events.py                   #     结构化事件协议（AgentEvent）
│   │   │   ├── llm_factory.py              #     strong/weak 分层 LLM 工厂（OpenRouter）
│   │   │   ├── middleware.py               #   Token 裁剪中间件 + 工具调用连续性修复
│   │   │   ├── react_runtime.py            #     ReAct 运行时（记忆召回注入 + episodic 写入）
│   │   │   ├── plan_execute_runtime.py     #     Plan-Execute 运行时包装
│   │   │   └── parallel_runtime.py         #     多 Agent 并行运行时包装
│   │   ├── aiops/                          #   AIOps Agent（Plan-Execute-Replan）
│   │   │   ├── state.py                    #     状态定义（含诊断事件流）
│   │   │   ├── planner.py                  #     规划器（路由 + KG + 增强检索 + 上下文组装）
│   │   │   ├── executor.py                 #     执行器（工具调用 + Agent Rules）
│   │   │   ├── replanner.py                #     重规划器（continue/replan/respond）
│   │   │   └── utils.py                    #     工具函数
│   │   └── multi/                          #   多 Agent 并行诊断
│   │       ├── coordinator.py              #     Coordinator 调度器
│   │       ├── specialists.py              #     3 个专业 Agent（日志/指标/知识）
│   │       └── synthesizer.py              #     Synthesizer 交叉验证
│   ├── tools/                              # Agent 工具
│   │   ├── knowledge_tool.py               #   知识检索（Rewrite + 向量 + BM25 + RRF + Rerank）
│   │   ├── kg_tool.py                      #   知识图谱查询工具
│   │   └── time_tool.py                    #   时间工具
│   ├── models/                             # 数据模型
│   │   ├── aiops.py                        #   AIOps 请求/响应模型
│   │   ├── diagnosis_report.py             #   结构化诊断报告 + 事件流 + 反馈模型
│   │   ├── document.py                     #   文档模型
│   │   ├── request.py                      #   请求模型
│   │   └── response.py                     #   响应模型
│   ├── harness/                            # Harness 约束层
│   │   └── agent_rules.py                  #   运维诊断规则库（10 通用 + 10 专项）
│   ├── eval/                               # 评测模块
│   │   └── evaluator.py                    #   端到端评测框架（5 场景 × 4 维度）
│   ├── core/                               # 核心组件
│   │   ├── context_engine.py               #   Token 预算上下文引擎（Packet/配额/压缩）
│   │   ├── token_budget.py                 #   Token 估算与预算降级
│   │   ├── llm_factory.py                  #   LLM 模型工厂
│   │   └── milvus_client.py                #   Milvus 客户端管理器
│   └── utils/
│       └── logger.py                       #   日志配置
├── mcp_servers/                            # MCP 外部工具服务器
│   ├── cls_server.py                       #   腾讯云日志查询 MCP Server
│   └── monitor_server.py                   #   监控指标查询 MCP Server
├── static/                                 # 前端静态文件
│   ├── index.html                          #   单页应用 HTML
│   ├── app.js                              #   前端逻辑（含 KG 可视化 + 反馈按钮）
│   └── styles.css                          #   样式
├── aiops-docs/                             # 运维知识库文档
├── vector-database.yml                     # Milvus Docker Compose
├── pyproject.toml                          # 项目配置 + 依赖
└── .env                                    # 环境变量
```

## ⚙️ 配置说明

通过 `.env` 文件配置：

```bash
# LLM 接入（必填）：通过 OpenRouter 的 OpenAI 兼容端点调用
# Key 管理：https://openrouter.ai/settings/keys （建议设置消费上限并定期轮换）
OPENROUTER_API_KEY=sk-or-v1-xxxx

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# RAG 配置
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# Embedding / Rerank（本地运行，无需 API Key；首次运行自动下载模型）
# EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5      # 1024 维，中文运维文档检索
# EMBEDDING_DEVICE=                            # 留空自动：mps > cpu
# RERANK_ENABLED=True
# RERANK_MODEL=BAAI/bge-reranker-base

# 弱模型层（路由/改写等轻任务，可选免费档）
# LLM_BACKUP_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

# 长期记忆（默认开启；关闭后所有记忆操作为无副作用空操作，行为与无记忆版本一致）
# MEMORY_ENABLED=True
# MEMORY_DB_PATH=data/memory.db
# MEMORY_RECALL_K=5                        # 每次召回上限
# MEMORY_MIN_IMPORTANCE=0.2                # 重要性下限过滤
# MEMORY_DECAY_LAMBDA=0.05                 # 时近衰减 λ（每天）
# MEMORY_WEIGHT_RELEVANCE=0.6              # 打分权重：相关性
# MEMORY_WEIGHT_IMPORTANCE=0.25            #            重要性
# MEMORY_WEIGHT_RECENCY=0.15               #            时近性
# MEMORY_CONSOLIDATE_THRESHOLD=0.85        # 巩固聚类余弦阈值

# 上下文引擎
# CONTEXT_TOKEN_BUDGET=6000                # 装配总预算（token）
# CONTEXT_HISTORY_BUDGET=2400              # 对话历史裁剪预算
```

> **更换 Embedding 模型后**，旧向量全部失效，需执行 `make reindex-drop` 重建索引；
> 仅重灌文档用 `make reindex`。

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 两种诊断模式

**模式 1: Plan-Execute-Replan** (`POST /api/aiops`)
```
查询路由 → Planner（意图分类 + KG查询 + 增强检索 → 生成计划）
         → Executor（逐步执行，调用 MCP 工具，遵循 Harness 诊断规则）
         → Replanner（评估：继续 / 调整计划 / 生成报告）
         → SSE 流式推送诊断过程 + 结构化报告
```

**模式 2: 多 Agent 并行诊断** (`POST /api/multi-diagnose`)
```
Coordinator（分析告警，并行分发）
    ├→ LogAnalystAgent（MCP:CLS 日志分析）
    ├→ MetricInspectorAgent（MCP:Monitor 指标分析）
    └→ KnowledgeRetrieverAgent（KG + RAG 知识检索）
→ Synthesizer（交叉验证 → 根因定位 → 结构化报告）
```

## 📝 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Docker 容器
make down              # 停止 Docker 容器

# 代码质量
make format            # 格式化代码
make lint              # 代码检查
```


## 🐛 常见问题

### Windows 环境问题

#### 1. `make` 命令不可用
Windows 不支持 `make` 命令，请使用提供的批处理脚本：
```powershell
# 启动服务
.\start-windows.bat

# 停止服务
.\stop-windows.bat
```

#### 2. PowerShell 执行策略限制
如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 错误：
```powershell
# 临时允许脚本执行（管理员权限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或者使用 CMD 而不是 PowerShell
cmd
.\start-windows.bat
```

#### 3. 端口被占用（Windows）
```powershell
# 查看占用端口的进程
netstat -ano | findstr :9900

# 结束进程（替换 PID 为实际进程 ID）
taskkill /F /PID <PID>
```

### 通用问题

### API Key 错误
```bash
# 检查环境变量
cat .env | grep OPENROUTER_API_KEY    # Linux/macOS
type .env | findstr OPENROUTER_API_KEY  # Windows
```

### Milvus 连接失败
```bash
# 确保本机有 Docker 服务并且已经启动（可以使用 Docker Desktop）

# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus（使用 docker compose）
docker compose -f vector-database.yml restart

# 或者重启单个服务
docker compose -f vector-database.yml restart standalone
```

### 服务无法启动

**Linux/macOS:**
```bash
# 查看服务日志
tail -f logs/app_$(date +%Y-%m-%d).log  # FastAPI 主服务（Loguru 日志）
tail -f mcp_cls.log                      # CLS MCP 服务
tail -f mcp_monitor.log                  # Monitor MCP 服务

# 检查端口占用
lsof -i :9900  # FastAPI
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

**Windows:**
```powershell
# 查看服务日志（获取今天的日期）
$today = Get-Date -Format "yyyy-MM-dd"
type logs\app_$today.log  # FastAPI 主服务（Loguru 日志）
type mcp_cls.log          # CLS MCP 服务
type mcp_monitor.log      # Monitor MCP 服务

# 或者查看最新的日志文件
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# 检查端口占用
netstat -ano | findstr :9900  # FastAPI
netstat -ano | findstr :8003  # CLS MCP
netstat -ano | findstr :8004  # Monitor MCP
```

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [OpenRouter](https://openrouter.ai/)
- [MCP 协议](https://modelcontextprotocol.io/)

## 📄 许可证
author： chief

MIT License
