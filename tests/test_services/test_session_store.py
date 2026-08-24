"""会话读写服务测试（P5-c）

转换逻辑自 react_runtime 机械迁出：此处钉住双形态检查点收窄、
系统消息过滤、时间戳兜底与三种清空容错语义。
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.session_store import SessionStore


class FakeCheckpointer:
    """模拟 MemorySaver 的 get/delete_thread 接口"""

    def __init__(self, checkpoint=None, *, raise_on_get=False, raise_on_delete=False):
        self._checkpoint = checkpoint
        self._raise_on_get = raise_on_get
        self._raise_on_delete = raise_on_delete
        self.deleted: list[str] = []

    def get(self, config):
        if self._raise_on_get:
            raise RuntimeError("boom")
        return self._checkpoint

    def delete_thread(self, thread_id):
        if self._raise_on_delete:
            raise RuntimeError("boom")
        self.deleted.append(thread_id)


def _checkpoint_of(*messages):
    return SimpleNamespace(checkpoint={"channel_values": {"messages": list(messages)}})


class TestReadMessages:
    def test_converts_roles_and_filters_system(self):
        cp = _checkpoint_of(
            SystemMessage(content="系统提示词"),
            HumanMessage(content="CPU 打满怎么查"),
            AIMessage(content="先看 top"),
        )
        history = SessionStore(FakeCheckpointer(cp)).read_messages("s1")

        assert [m["role"] for m in history] == ["user", "assistant"]
        assert [m["content"] for m in history] == ["CPU 打满怎么查", "先看 top"]
        assert all(m["timestamp"] for m in history)

    def test_plain_tuple_checkpoint_form_supported(self):
        """旧式普通元组形态 ((checkpoint_dict,), ...) 同样可读"""
        raw = ({"channel_values": {"messages": [HumanMessage(content="hi")]}},)
        history = SessionStore(FakeCheckpointer(raw)).read_messages("s1")
        assert history[0]["role"] == "user"

    def test_no_checkpoint_returns_empty(self):
        assert SessionStore(FakeCheckpointer(None)).read_messages("s1") == []

    def test_broken_checkpointer_returns_empty_not_raise(self):
        store = SessionStore(FakeCheckpointer(raise_on_get=True))
        assert store.read_messages("s1") == []

    def test_real_memory_saver_checkpoint_dict_form(self):
        """评审修复回归：真实 MemorySaver.get() 返回裸 Checkpoint dict
        （顶层即 channel_values），此前双形态收窄双双失配致历史恒空"""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph

        class _St(dict):
            pass

        def _graph():
            from typing import Annotated, TypedDict

            from langgraph.graph.message import add_messages

            class St(TypedDict):
                messages: Annotated[list, add_messages]

            def node(state):
                return {"messages": [("assistant", "答")]}

            g = StateGraph(St)
            g.add_node("n", node)
            g.set_entry_point("n")
            g.add_edge("n", END)
            return g

        cp = MemorySaver()
        app = _graph().compile(checkpointer=cp)
        app.invoke(
            {"messages": [("user", "问")]}, {"configurable": {"thread_id": "t-real"}}
        )

        history = SessionStore(cp).read_messages("t-real")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "问"


class TestClearSemantics:
    async def test_clear_success_and_failure(self):
        ok = SessionStore(FakeCheckpointer())
        assert ok.clear("a") is True
        assert ok.checkpointer.deleted == ["a"]

        bad = SessionStore(FakeCheckpointer(raise_on_delete=True))
        assert bad.clear("b") is False

    async def test_clear_best_effort_never_raises(self):
        bad = SessionStore(FakeCheckpointer(raise_on_delete=True))
        bad.clear_best_effort("c")  # 不抛即通过


class TestRuntimeDelegation:
    def test_react_runtime_snapshot_delegates_to_store(self):
        """runtime.snapshot 的消息视图与 SessionStore 直读一致；
        构造后整体替换 checkpointer 也必须生效（视图跟随当前实例）"""
        from app.agent.runtime.react_runtime import ReActRuntime

        rt = ReActRuntime(tools=[], system_prompt="测试")
        rt.checkpointer = FakeCheckpointer(
            _checkpoint_of(HumanMessage(content="q"), AIMessage(content="a"))
        )

        snap = rt.snapshot("sess-9")
        assert [m["role"] for m in snap["messages"]] == ["user", "assistant"]

    def test_react_reset_delegates_to_store(self):
        from app.agent.runtime.react_runtime import ReActRuntime

        rt = ReActRuntime(tools=[], system_prompt="测试")
        fake = FakeCheckpointer()
        rt.checkpointer = fake

        assert rt.reset("sess-1") is True
        assert fake.deleted == ["sess-1"]
