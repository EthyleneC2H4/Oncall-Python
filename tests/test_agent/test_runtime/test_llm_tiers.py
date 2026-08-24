"""运行时基础组件测试：注册表 + LLM 分层工厂缓存语义"""

import pytest

from app.agent.runtime.base import AgentRuntime, RuntimeRegistry
from app.agent.runtime.llm_factory import TieredLLM
from app.core.llm_factory import LLMFactory


class FakeRuntime(AgentRuntime):
    def __init__(self, name: str):
        self.name = name

    async def run(self, task, session_id="default"):  # pragma: no cover - 接口桩
        raise NotImplementedError

    def snapshot(self, session_id):  # pragma: no cover - 接口桩
        return {}

    def reset(self, session_id):  # pragma: no cover - 接口桩
        return True


class TestRuntimeRegistry:
    def test_register_and_get(self):
        registry = RuntimeRegistry()
        rt = FakeRuntime("react")
        registry.register(rt)

        assert registry.get("react") is rt

    def test_get_missing_returns_none(self):
        registry = RuntimeRegistry()
        assert registry.get("nope") is None

    def test_same_name_overwrites(self):
        """同名注册覆盖（服务重建实例后新实例生效）"""
        registry = RuntimeRegistry()
        rt1 = FakeRuntime("plan_execute")
        rt2 = FakeRuntime("plan_execute")
        registry.register(rt1)
        registry.register(rt2)

        assert registry.get("plan_execute") is rt2

    def test_names_lists_all(self):
        registry = RuntimeRegistry()
        registry.register(FakeRuntime("a"))
        registry.register(FakeRuntime("b"))

        assert sorted(registry.names()) == ["a", "b"]


@pytest.fixture(autouse=True)
def _isolate_llm_instance_cache():
    """隔离 LLMFactory 类级实例缓存，避免跨测试污染"""
    saved = dict(LLMFactory._instances)
    LLMFactory._instances.clear()
    yield
    LLMFactory._instances.clear()
    LLMFactory._instances.update(saved)


class TestTieredLLMCacheIdentity:
    """LLM 实例缓存恒等（P1.1 目标：每节点/每步不再重复构造客户端）"""

    def test_same_args_return_cached_instance(self):
        llm_a = LLMFactory.create_chat_model(model="test/model-a", temperature=0.0)
        llm_b = LLMFactory.create_chat_model(model="test/model-a", temperature=0.0)

        assert llm_a is llm_b

    def test_different_temperature_different_instance(self):
        llm_a = LLMFactory.create_chat_model(model="test/model-a", temperature=0.0)
        llm_b = LLMFactory.create_chat_model(model="test/model-a", temperature=0.7)

        assert llm_a is not llm_b

    def test_custom_api_key_bypasses_cache(self):
        """注入自定义 key 的实例不进缓存（测试隔离语义）"""
        base = LLMFactory.create_chat_model(model="test/model-a", temperature=0.0)
        injected1 = LLMFactory.create_chat_model(
            model="test/model-a", temperature=0.0, api_key="sk-test-1"
        )
        injected2 = LLMFactory.create_chat_model(
            model="test/model-a", temperature=0.0, api_key="sk-test-2"
        )

        assert injected1 is not base
        assert injected1 is not injected2
        # 注入实例不应污染默认缓存
        assert LLMFactory.create_chat_model(model="test/model-a", temperature=0.0) is base

    def test_tiered_strong_uses_primary_model(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.llm_factory.config.rag_model", "primary/strong-model"
        )
        llm = TieredLLM.strong(temperature=0.0)
        assert llm.model_name == "primary/strong-model"

    def test_tiered_weak_uses_backup_model(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.llm_factory.config.llm_backup_model", "backup/weak-model"
        )
        llm = TieredLLM.weak(temperature=0.5)
        assert llm.model_name == "backup/weak-model"

    def test_strong_weak_layers_are_distinct_instances(self, monkeypatch):
        monkeypatch.setattr("app.core.llm_factory.config.rag_model", "primary/m")
        monkeypatch.setattr("app.core.llm_factory.config.llm_backup_model", "backup/m")

        assert TieredLLM.strong() is not TieredLLM.weak()
