"""pytest 共享 fixtures 和配置

为 OnCall 项目提供测试基础设施：
- Mock 外部依赖（OpenRouter LLM、本地 Embedding、Milvus、MCP）
- 环境变量注入
- 隔离的审计日志和缓存实例
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ──────────────── 路径与环境 ────────────────


@pytest.fixture(autouse=True)
def _patch_project_root(monkeypatch):
    """确保所有路径计算基于项目根目录"""
    project_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(project_root)
    # 注入假 API Key：保证 ChatOpenAI 可构造（不发真实请求）
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key-for-testing")
    return project_root


@pytest.fixture(autouse=True)
def _isolated_memory_singleton(tmp_path, monkeypatch):
    """默认隔离长期记忆单例：临时库路径 + 确定性假嵌入器

    防止任何测试经 planner/react 等集成点触发真实 BGE 模型加载
    （会发起 HuggingFace 下载，拖垮甚至卡死整个测试会话），
    也避免测试污染仓库内 data/memory.db。
    专测记忆行为的用例自行注入 FakeEmbedder / 换 store，不受此影响。
    """
    import app.services.memory as memory_pkg
    from app.config import config

    svc = memory_pkg.memory_service
    saved_embedder = svc.embedder
    saved_store = svc._store

    class _StubEmbedder:
        """确定性伪向量嵌入器（无模型依赖，向量随文本长度稳定变化）"""

        def embed_query_safe(self, text: str) -> list[float]:
            return [float(len(text) % 7 + 1), 1.0, 0.0]

    if saved_store is not None:
        try:
            saved_store.close()
        except Exception:  # noqa: BLE001 - 测试隔离尽力而为
            pass

    monkeypatch.setattr(config, "memory_db_path", str(tmp_path / "memory.db"))
    svc.embedder = _StubEmbedder()
    svc._store = None  # 强制下次使用时按临时路径重建
    yield
    svc.embedder = saved_embedder
    svc._store = None


# ──────────────── 外部服务 Mock ────────────────


def make_fake_llm(response=None):
    """构造 LLMFactory.create_chat_model 的返回替身

    覆盖三种调用路径：
    - llm.ainvoke / llm.invoke          （直接调用）
    - llm.with_structured_output(...).ainvoke（结构化输出链，planner/replanner 使用）
    - llm.bind_tools(...).ainvoke       （工具绑定，executor/agent 使用）
    """
    default_response = MagicMock(
        content="这是一个测试回答",
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
    )
    llm = MagicMock(name="FakeChatModel")
    llm.ainvoke = AsyncMock(return_value=response if response is not None else default_response)
    llm.invoke = MagicMock(return_value=MagicMock(content="测试回答"))

    structured = MagicMock(name="FakeStructuredLLM")
    structured.ainvoke = llm.ainvoke  # 共享响应序列
    structured.invoke = llm.invoke
    llm.with_structured_output = MagicMock(return_value=structured)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_llm_factory():
    """Mock LLMFactory.create_chat_model，所有模型请求均返回固定替身"""
    fake_llm = make_fake_llm()
    with patch("app.core.llm_factory.LLMFactory.create_chat_model", return_value=fake_llm) as m:
        m.fake_llm = fake_llm  # 测试可进一步定制响应
        yield m


@pytest.fixture
def mock_embeddings():
    """Mock 本地 BGE Embedding 服务（避免下载/加载真实模型）"""
    fake_vector = [0.1] * 1024
    with (
        patch(
            "app.services.vector_embedding_service.vector_embedding_service.embed_query",
            return_value=fake_vector,
        ),
        patch(
            "app.services.vector_embedding_service.vector_embedding_service.embed_query_safe",
            return_value=fake_vector,
        ),
        patch(
            "app.services.vector_embedding_service.vector_embedding_service.embed_documents",
            return_value=[fake_vector],
        ),
    ):
        yield


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client（OpenRouter OpenAI 兼容端点）"""
    with patch("openai.OpenAI", autospec=True) as mock_cls:
        instance = mock_cls.return_value
        instance.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1024)]
        )
        yield mock_cls


@pytest.fixture
def mock_milvus():
    """Mock Milvus 客户端"""
    with patch("app.core.milvus_client.MilvusClient") as mock_cls:
        instance = mock_cls.return_value
        instance.list_collections.return_value = ["oncall_knowledge"]
        instance.search.return_value = [[]]
        instance.describe_collection.return_value = {"num_entities": 0}
        yield mock_cls


@pytest.fixture
def mock_mcp_client():
    """Mock MCP 客户端"""
    with patch("app.agent.mcp_client.get_mcp_client_with_retry", new_callable=AsyncMock) as mock:
        instance = AsyncMock()
        instance.get_tools = AsyncMock(return_value=[])
        instance.cleanup = AsyncMock()
        mock.return_value = instance
        yield mock


# ──────────────── 隔离的审计日志 ────────────────


@pytest.fixture
def temp_audit_dir(monkeypatch, tmp_path):
    """使用临时目录存放审计日志"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr("app.core.audit.LOG_DIR", str(log_dir))
    return log_dir


# ──────────────── 隔离的缓存 ────────────────


@pytest.fixture
def clean_cache():
    """干净的 TTLCache 实例"""
    from app.core.cache import TTLCache

    cache = TTLCache(name="test", maxsize=10, ttl_seconds=60)
    return cache


# ──────────────── 数据集 Fixtures ────────────────


@pytest.fixture
def sample_diagnostic_case():
    """单个诊断用例"""
    return {
        "id": "TEST001",
        "category": "easy",
        "query": "CPU使用率100%，服务无响应",
        "expected_intent": "DIAGNOSTIC",
        "expected_root_causes": ["MemoryLeak", "内存泄漏"],
        "expected_docs": ["cpu_high_usage.md"],
        "expected_answer_contains": ["CPU", "内存"],
        "tags": ["cpu"],
        "reference": "CPU 100% 通常由内存泄漏或死循环引起，建议检查进程状态和内存使用。",
    }


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """临时数据集目录"""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    diagnostic = [
        {
            "id": "TEST001",
            "category": "easy",
            "query": "CPU使用率100%",
            "expected_intent": "DIAGNOSTIC",
            "expected_root_causes": ["MemoryLeak"],
            "expected_docs": ["cpu_high_usage.md"],
            "expected_answer_contains": ["CPU"],
            "tags": ["cpu"],
            "reference": "测试参考回答。",
        }
    ]
    negative = [
        {
            "id": "NEG_TEST001",
            "category": "chitchat",
            "query": "你好",
            "expected_intent": "CHITCHAT",
            "expected_root_causes": [],
            "expected_docs": [],
            "tags": ["greeting"],
            "reference": "你好，我是运维助手。",
        }
    ]
    with open(datasets_dir / "diagnostic_cases.json", "w") as f:
        json.dump(diagnostic, f, ensure_ascii=False)
    with open(datasets_dir / "negative_cases.json", "w") as f:
        json.dump(negative, f, ensure_ascii=False)
    return datasets_dir


# ──────────────── FastAPI TestClient ────────────────


@pytest.fixture
def test_app(mock_llm_factory, mock_embeddings, mock_milvus, mock_mcp_client):
    """创建测试用的 FastAPI app（mock 所有外部依赖）"""
    from fastapi.testclient import TestClient

    from app.main import app

    # 禁用 lifespan 以避免连接外部服务
    app.router.lifespan = None
    client = TestClient(app)
    return client


# ──────────────── 通用工具 ────────────────


@pytest.fixture
def assert_json_ok():
    """断言响应为 JSON 且 code=200"""

    def _assert(response, status_code=200):
        assert response.status_code == status_code
        data = response.json()
        assert data.get("code") == 200
        return data

    return _assert
