"""routes TDD:SSE 入口(图版)、两个动作接口(转人工 / 建工单)。"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent import graph as graph_module
from app.agent.graph import AgentDeps
from app.core.tools.registry import ToolRegistry
from app.db import get_session
from app.main import app
from app.models.ticket import Ticket
from app.services import agent_chat


class IntentModel:
    def __init__(self, intent="物流"):
        self.intent = intent

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"intent": "%s", "confidence": 0.9}' % self.intent)


class CheckModel:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"useful": true, "reason": "ok"}')


class FakeAgentModel:
    def __init__(self, with_tool=False):
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
                    {"name": "fake_logistics", "args": '{"order_id": "1001"}', "id": "call_1", "index": 0}
                ],
            )
            return
        yield AIMessageChunk(content="已为您查到：包裹在北京转运中心")


@tool
def fake_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


@pytest.fixture
async def client(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def stub_deps(monkeypatch):
    """把服务的生产依赖换成假模型;返回设置意图的函数。"""

    def _install(intent="物流", with_tool=False):
        reg = ToolRegistry()
        reg.register(fake_logistics)
        deps = AgentDeps(
            model=FakeAgentModel(with_tool=with_tool),
            registry=reg,
            session_factory=None,
            intent_model=IntentModel(intent),
            check_model=CheckModel(),
            kb_min_score=0.45,
        )
        monkeypatch.setattr(agent_chat, "build_default_deps", lambda _sf: deps)
        return deps

    return _install


async def test_chat_stream_endpoint_returns_event_stream(client, stub_deps):
    stub_deps("闲聊")
    resp = client.post("/api/v1/chat/stream", json={"session_id": "t1", "message": "你好"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in resp.text
    assert "event: done" in resp.text
    assert "event: node" in resp.text  # 节点可观测


async def test_empty_message_rejected(client):
    resp = client.post("/api/v1/chat/stream", json={"session_id": "t1", "message": "   "})
    assert resp.status_code == 422


async def test_tool_events_appear_in_stream(client, stub_deps):
    stub_deps("物流", with_tool=True)
    resp = client.post(
        "/api/v1/chat/stream", json={"session_id": "t1", "message": "订单1001的物流到哪了"}
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: tool_start" in body
    assert "fake_logistics" in body
    assert "event: tool_end" in body
    assert body.index("event: tool_start") < body.index("event: tool_end")
    assert body.index("event: tool_end") < body.index("event: done")


async def test_complaint_stream_carries_two_actions(client, stub_deps):
    stub_deps("投诉")
    resp = client.post("/api/v1/chat/stream", json={"session_id": "t1", "message": "我要投诉"})
    assert resp.status_code == 200
    assert "transfer_human" in resp.text
    assert "create_ticket" in resp.text


async def test_transfer_endpoint_returns_notice_and_persists(client, engine, session):
    from app.repositories import conversation_repo, message_repo

    conv = await conversation_repo.get_or_create(session, None, "u1")
    resp = client.post("/api/v1/chat/actions/transfer", json={"conversation_id": conv.id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["notice"] == "已转接人工客服"
    assert "客服小猫" in body["greeting"]

    msgs = await message_repo.list_by_conversation(session, conv.id)
    assert msgs and "客服小猫" in (msgs[-1].content or "")


async def test_transfer_does_not_create_ticket(client, session, engine):
    from app.repositories import conversation_repo

    conv = await conversation_repo.get_or_create(session, None, "u2")
    client.post("/api/v1/chat/actions/transfer", json={"conversation_id": conv.id})
    rows = (await session.execute(select(Ticket))).scalars().all()
    assert rows == []  # 转人工不建单:两件事,互不绑定


async def test_transfer_unknown_conversation_404(client):
    resp = client.post("/api/v1/chat/actions/transfer", json={"conversation_id": 999999})
    assert resp.status_code == 404


async def test_create_ticket_writes_ticket_row(client, monkeypatch, engine, session):
    """建工单走第 2 章的工具,不直接写 repo。"""
    from app.core.tools.business_tools import build_db_tools
    from app.repositories import conversation_repo, message_repo

    maker = async_sessionmaker(engine, expire_on_commit=False)
    reg = ToolRegistry()
    for t in build_db_tools(maker, retriever=None):
        reg.register(t)
    monkeypatch.setattr("app.api.routes.build_default_registry", lambda: reg)

    conv = await conversation_repo.get_or_create(session, None, "u3")
    await message_repo.add_message(session, conv.id, role="user", content="你们发错货了两次,我要投诉")

    resp = client.post(
        "/api/v1/chat/actions/create-ticket", json={"conversation_id": conv.id}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["ticket_no"]
    # 描述缺省取最近一条用户原话
    assert "发错货" in body["description"]

    rows = (await session.execute(select(Ticket))).scalars().all()
    assert len(rows) == 1
    assert rows[0].ticket_no == body["ticket_no"]


async def test_create_ticket_without_any_user_message_is_rejected(client, monkeypatch, engine, session):
    from app.core.tools.business_tools import build_db_tools
    from app.repositories import conversation_repo

    maker = async_sessionmaker(engine, expire_on_commit=False)
    reg = ToolRegistry()
    for t in build_db_tools(maker, retriever=None):
        reg.register(t)
    monkeypatch.setattr("app.api.routes.build_default_registry", lambda: reg)

    conv = await conversation_repo.get_or_create(session, None, "u4")
    resp = client.post("/api/v1/chat/actions/create-ticket", json={"conversation_id": conv.id})
    assert resp.status_code == 422


async def test_create_ticket_rejects_bad_type(client):
    resp = client.post(
        "/api/v1/chat/actions/create-ticket",
        json={"conversation_id": 1, "ticket_type": "随便"},
    )
    assert resp.status_code == 422
