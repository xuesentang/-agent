"""Langfuse 接入的单测:工厂四种降级、metadata 组装、检索包装、span 树。

span 树用例不联网:给 SDK 换一个内存 span exporter,拿假模型把真聊天链路
(`chat_with_agent` + 真图)跑一遍,断言 trace 结构——根 span、图/节点嵌套、
generation(带 prompt/completion/usage)、工具 span、意图元数据、abstain tag。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from app.agent import graph as graph_module
from app.agent.graph import AgentDeps
from app.config import get_settings
from app.core.tools.registry import ToolRegistry
from app.observability import tracing


class ListExporter(SpanExporter):
    """把 span 攒在内存里,不断网、不落盘。"""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class ProbeChat(BaseChatModel):
    """真 BaseChatModel:走 LangChain 回调,trace 里才有 generation span。"""

    reply: str = "{}"

    @property
    def _llm_type(self) -> str:
        return "probe-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        usage = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        msg = AIMessage(content=self.reply, usage_metadata=usage)
        gen = ChatGeneration(message=msg, generation_info={"usage_metadata": usage})
        return ChatResult(
            generations=[gen],
            llm_output={"token_usage": usage, "model_name": "probe-model"},
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


class ProbeStream:
    """有工具就走一步,没有就直接收敛。"""

    def __init__(self, with_tool=True):
        self.with_tool = with_tool
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls += 1
        if self.with_tool and self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "fake_order", "args": '{"order_id": "1001"}', "id": "c1", "index": 0}
                ],
            )
            return
        for t in ("好的,", "已为您查到。"):
            yield AIMessageChunk(content=t)


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return f"订单 {order_id} 已发货"


@pytest.fixture
def langfuse_env(monkeypatch):
    """按 key 打开 Langfuse 环境。每个用例换 key——SDK 的 client 实例按
    public_key 全局复用,同 key 会拿到上一个用例的 client(和它的 exporter);
    用例结束后把新建的实例摘掉,否则 keyless get_client(observe 用)会因为
    多实例返回禁用 client。"""

    def _enable(key: str) -> None:
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", key)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
        get_settings.cache_clear()

    from langfuse._client.resource_manager import LangfuseResourceManager

    before = set(LangfuseResourceManager._instances)
    yield _enable
    get_settings.cache_clear()
    for key in set(LangfuseResourceManager._instances) - before:
        LangfuseResourceManager._instances.pop(key, None)


def _probe_client(key: str, exporter: ListExporter):
    """带内存 exporter 的真 client——span 落在 exporter 里,不断网。"""
    from langfuse import Langfuse

    return Langfuse(
        public_key=key,
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracing_enabled=True,
    )


def _patch_ready(monkeypatch, key: str, exporter: ListExporter) -> None:
    """让工厂走通:host 检查打桩 + client 用内存 exporter。"""
    monkeypatch.setattr(tracing, "_probe_host", lambda host, timeout=3.0: True)
    monkeypatch.setattr(tracing, "_build_client", lambda settings: _probe_client(key, exporter))


# ------------------------------------------------------------ 工厂四种降级
def test_callbacks_empty_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-t1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    get_settings.cache_clear()
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_empty_when_public_key_missing(monkeypatch):
    langfuse_env_off = {"LANGFUSE_ENABLED": "true", "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "sk-lf-test"}
    for k, v in langfuse_env_off.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_empty_when_secret_key_missing(monkeypatch):
    for k, v in {"LANGFUSE_ENABLED": "true", "LANGFUSE_PUBLIC_KEY": "pk-lf-t3", "LANGFUSE_SECRET_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_empty_when_sdk_unavailable(monkeypatch, langfuse_env):
    langfuse_env("pk-lf-t4")
    monkeypatch.setattr(tracing, "_load_sdk", lambda: None)
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_empty_when_host_unreachable(monkeypatch, langfuse_env):
    langfuse_env("pk-lf-t5")
    monkeypatch.setattr(tracing, "_probe_host", lambda host, timeout=3.0: False)
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_empty_when_client_init_fails(monkeypatch, langfuse_env):
    langfuse_env("pk-lf-t6")

    def _boom(settings):
        raise RuntimeError("bad credentials")

    monkeypatch.setattr(tracing, "_probe_host", lambda host, timeout=3.0: True)
    monkeypatch.setattr(tracing, "_build_client", _boom)
    assert tracing.get_langfuse_callbacks() == []


def test_callbacks_returns_handler_when_ready(monkeypatch, langfuse_env):
    from langfuse.langchain import CallbackHandler

    langfuse_env("pk-lf-t7")
    _patch_ready(monkeypatch, "pk-lf-t7", ListExporter())
    cbs = tracing.get_langfuse_callbacks()
    assert len(cbs) == 1
    assert isinstance(cbs[0], CallbackHandler)


# ------------------------------------------------------------ metadata 组装
def test_build_trace_metadata_normal():
    meta = tracing.build_trace_metadata(conversation_id=7, session_id="sess-abc")
    assert meta == {"langfuse_trace_name": "chat:7", "langfuse_session_id": "sess-abc"}


def test_build_trace_metadata_drops_invalid_session():
    # 非 ASCII / 超长 / 空串:Langfuse 只会告警丢弃,这里先过滤掉
    assert "langfuse_session_id" not in tracing.build_trace_metadata(
        conversation_id=1, session_id="会话-1"
    )
    assert "langfuse_session_id" not in tracing.build_trace_metadata(
        conversation_id=1, session_id="x" * 201
    )
    assert "langfuse_session_id" not in tracing.build_trace_metadata(
        conversation_id=1, session_id=""
    )
    # trace_name 与 session 无关,永远在
    assert tracing.build_trace_metadata(conversation_id=1, session_id="")["langfuse_trace_name"] == "chat:1"


# ------------------------------------------------------------ 补写入口降级
def test_record_intent_route_noop_when_disabled():
    tracing.record_intent_route(None, intent="物流", route="business", confidence=0.9)


def test_mark_abstain_noop_when_disabled():
    tracing.mark_abstain(None)


# ------------------------------------------------------------ 检索包装
def test_traced_identity_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()

    async def fn():
        return None

    from app.rag import retriever as retriever_module

    assert retriever_module._traced(fn) is fn


async def test_traced_wraps_retrieve_and_emits_span(monkeypatch, langfuse_env):
    """启用时检索被 observe 包一层;调用后 trace 里有 retriever span。"""
    from app.rag import retriever as retriever_module

    langfuse_env("pk-lf-t8")
    exporter = ListExporter()
    _patch_ready(monkeypatch, "pk-lf-t8", exporter)

    async def stub(**kwargs):
        return []

    wrapped = retriever_module._traced(stub)
    assert wrapped is not stub

    with tracing.open_chat_trace(conversation_id=99, session_id="sess-x"):
        await wrapped(query="退货运费", top_k=3)
    tracing.flush_traces()

    spans = [s for s in exporter.spans if s.name == "rag.retrieve"]
    assert len(spans) == 1
    assert spans[0].attributes["langfuse.observation.type"] == "retriever"


# ------------------------------------------------------------ span 树(真链路)
def _deps(intent_reply: str, score: float, monkeypatch, with_tool: bool = True) -> AgentDeps:
    reg = ToolRegistry()
    reg.register(fake_order)

    async def fake_retrieve(**kwargs):
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(
                    id=1,
                    section_path="退货退款政策 > 运费规则",
                    questions="退货运费谁承担",
                    answer="七日无理由退货的寄回运费由消费者承担。",
                ),
                score=score,
            )
        ]

    monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)
    return AgentDeps(
        model=ProbeStream(with_tool=with_tool),
        registry=reg,
        session_factory=None,
        intent_model=ProbeChat(reply=intent_reply),
        understand_model=ProbeChat(
            reply='{"resolved": "邮费是多少", "canonical": "运费", "changed": false}'
        ),
        check_model=ProbeChat(reply='{"useful": true, "reason": "probe"}'),
        kb_min_score=0.45,
    )


async def _chat(session, deps, text, **kw):
    from app.services.agent_chat import chat_with_agent

    return [e async for e in chat_with_agent("sess-1", text, session=session, deps=deps, **kw)]


def _attrs(span, prefix: str) -> dict:
    return {k: v for k, v in span.attributes.items() if k.startswith(prefix)}


async def test_trace_tree_covers_graph_generations_tools_and_metadata(session, monkeypatch, langfuse_env):
    """一轮对话 = 一条 trace:根 span → 图 → 节点 → generation/工具,意图进元数据。"""
    exporter = ListExporter()
    langfuse_env("pk-lf-t9")
    _patch_ready(monkeypatch, "pk-lf-t9", exporter)

    events = await _chat(
        session,
        _deps('{"intent": "商品咨询", "confidence": 0.91}', 0.62, monkeypatch),
        "邮费是多少",
    )
    done = events[-1].data
    assert done["finish_reason"] == "stop"
    assert done["intent"] == "商品咨询"
    assert done["route"] == "knowledge"

    conv_id = done["conversation_id"]
    roots = [s for s in exporter.spans if s.parent is None and s.name == f"chat:{conv_id}"]
    assert len(roots) == 1, "一轮对话一条 trace,根 span 只能有一个"
    root = roots[0]

    # 会话归组 + 意图/路由元数据(值是 JSON 编码的字符串)
    assert root.attributes["session.id"] == "sess-1"
    meta = _attrs(root, "langfuse.trace.metadata.")
    assert json.loads(meta["langfuse.trace.metadata.intent"]) == "商品咨询"
    assert json.loads(meta["langfuse.trace.metadata.route"]) == "knowledge"

    # 图执行嵌在同一 trace 里:根 → LangGraph → 各节点
    children = [s for s in exporter.spans if s.parent and s.parent.span_id == root.context.span_id]
    graph_span = next(s for s in children if s.name == "LangGraph")
    node_names = {
        json.loads(s.attributes["langfuse.observation.metadata.langgraph_node"])
        for s in exporter.spans
        if s.parent
        and s.parent.span_id == graph_span.context.span_id
        and "langfuse.observation.metadata.langgraph_node" in s.attributes
    }
    assert {"classify_intent", "retrieve_knowledge", "confidence_gate", "agent", "log_trace"} <= node_names

    # generation:prompt / completion / usage 都在
    gens = [s for s in exporter.spans if s.attributes.get("langfuse.observation.type") == "generation"]
    assert gens
    with_io = [
        s
        for s in gens
        if "langfuse.observation.input" in s.attributes
        and "langfuse.observation.output" in s.attributes
    ]
    assert with_io, "generation 要带 prompt 和 completion"
    assert any(
        k.startswith("langfuse.observation.usage") for s in gens for k in s.attributes
    ), "generation 要带 usage"

    # 工具 span
    tools = [s for s in exporter.spans if s.attributes.get("langfuse.observation.type") == "tool"]
    assert [s.name for s in tools] == ["fake_order"]


async def test_abstain_turn_root_carries_abstain_tag(session, monkeypatch, langfuse_env):
    """拒答轮(finish_reason=abstain)根 span 打 abstain tag,UI 里可按 tag 过滤。"""
    exporter = ListExporter()
    langfuse_env("pk-lf-t10")
    _patch_ready(monkeypatch, "pk-lf-t10", exporter)

    events = await _chat(
        session,
        _deps('{"intent": "商品咨询", "confidence": 0.88}', 0.30, monkeypatch, with_tool=False),
        "W60 用的碳板是第几代",
    )
    done = events[-1].data
    assert done["finish_reason"] == "abstain"

    conv_id = done["conversation_id"]
    root = next(s for s in exporter.spans if s.parent is None and s.name == f"chat:{conv_id}")
    tags = root.attributes.get("langfuse.trace.tags") or ()
    assert "abstain" in tags


async def test_disabled_langfuse_leaves_no_spans(session, monkeypatch):
    """LANGFUSE_ENABLED=false:聊天正常,一个 span 都不产。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()

    events = await _chat(
        session,
        _deps('{"intent": "商品咨询", "confidence": 0.91}', 0.62, monkeypatch),
        "邮费是多少",
    )
    assert events[-1].data["finish_reason"] == "stop"
    assert tracing.get_langfuse_callbacks() == []
