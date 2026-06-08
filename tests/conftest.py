"""pytest 共享 fixtures 和配置

为 OnCall 项目提供测试基础设施：
- Mock 外部依赖（DashScope、Milvus、MCP）
- 环境变量注入
- 隔离的审计日志和缓存实例
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────── 路径与环境 ────────────────

@pytest.fixture(autouse=True)
def _patch_project_root(monkeypatch):
    """确保所有路径计算基于项目根目录"""
    project_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(project_root)
    # 模拟 os.environ 中不存在 DASHSCOPE_API_KEY 时 config 不会崩溃
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key-for-testing")
    return project_root


# ──────────────── 外部服务 Mock ────────────────

@pytest.fixture
def mock_chat_qwen():
    """Mock ChatQwen，返回固定回答"""
    with patch("langchain_qwq.ChatQwen", autospec=True) as mock_cls:
        instance = mock_cls.return_value
        instance.ainvoke = AsyncMock()
        instance.ainvoke.return_value = MagicMock(
            content="这是一个测试回答",
            usage_metadata={"input_tokens": 100, "output_tokens": 50},
        )
        instance.invoke = MagicMock(return_value=MagicMock(content="测试回答"))
        yield mock_cls


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client（DashScope 兼容模式）"""
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

    cache = TTLCache(max_size=10, ttl=60)
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
def test_app(mock_chat_qwen, mock_openai_client, mock_milvus, mock_mcp_client):
    """创建测试用的 FastAPI app（mock 所有外部依赖）"""
    from app.main import app
    from fastapi.testclient import TestClient

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
