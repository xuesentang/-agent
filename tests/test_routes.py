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


class UnderstandModel:
    """原样透传:接口测试不该关心理解质量。"""

    async def ainvoke(self, messages, **kwargs):
        prompt = str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1])
        question = prompt.rsplit("用户这句话:", 1)[-1].strip()
        return AIMessage(
            content='{"resolved": "%s", "canonical": "%s", "order_id": null, "changed": false}'
            % (question, question)
        )


class ExpandModel:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"queries": ["退货政策"]}')


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
            understand_model=UnderstandModel(),
            expand_model=ExpandModel(),
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


async def test_refund_reasons_endpoint_lists_fixed_categories(client):
    resp = client.get("/api/v1/refund/reasons")
    assert resp.status_code == 200
    reasons = resp.json()["reasons"]
    assert "质量问题" in reasons
    assert len(reasons) >= 6


async def test_refund_apply_writes_ticket_and_rejects_unknown_reason(client, monkeypatch, engine, session):
    from app.core.tools.business_tools import build_db_tools
    from app.repositories import conversation_repo

    maker = async_sessionmaker(engine, expire_on_commit=False)
    reg = ToolRegistry()
    for t in build_db_tools(maker, retriever=None):
        reg.register(t)
    monkeypatch.setattr("app.api.routes.build_default_registry", lambda: reg)

    conv = await conversation_repo.get_or_create(session, None, "u-refund")
    bad = client.post(
        "/api/v1/refund/apply",
        json={"conversation_id": conv.id, "order_id": "1001", "reason": "随便编的理由"},
    )
    assert bad.status_code == 422  # 原因必须来自固定类目

    ok = client.post(
        "/api/v1/refund/apply",
        json={"conversation_id": conv.id, "order_id": "1001", "reason": "质量问题"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "submitted"
    assert body["refund_no"]
    rows = (await session.execute(select(Ticket))).scalars().all()
    assert len(rows) == 1
    assert "质量问题" in (rows[0].description or "")


async def test_stream_accepts_order_id_slot(client, stub_deps, monkeypatch):
    """点选订单卡片后前端会回填 order_id,接口要接得住。"""
    from app.agent import refund_flow as refund_module

    async def no_hits(**kwargs):
        return []

    monkeypatch.setattr(refund_module, "retrieve", no_hits)
    stub_deps("退款退货")
    resp = client.post(
        "/api/v1/chat/stream",
        json={"session_id": "t1", "message": "这个能退吗", "order_id": "1001"},
    )
    assert resp.status_code == 200
    assert "event: done" in resp.text


# ---------------------------------------------------------------- CH07 只读会话接口


async def test_conversations_lists_new_first_with_preview(client, session):
    from app.repositories import conversation_repo, message_repo

    first = await conversation_repo.get_or_create(session, None, "u-side")
    await message_repo.add_message(session, first.id, role="user", content="邮费是多少")
    second = await conversation_repo.get_or_create(session, None, "u-side", force_new=True)
    await message_repo.add_message(session, second.id, role="user", content="订单 1001 到哪了")

    resp = client.get("/api/v1/conversations", params={"user_key": "u-side"})
    assert resp.status_code == 200
    items = resp.json()["conversations"]
    assert [c["conversation_id"] for c in items] == [second.id, first.id]  # 新的在前
    assert items[0]["preview"] == "订单 1001 到哪了"
    assert items[0]["summarized"] is False


async def test_conversations_marks_summarized(client, session):
    from app.repositories import conversation_repo, conversation_summary_repo

    conv = await conversation_repo.get_or_create(session, None, "u-side2")
    await conversation_summary_repo.advance_anchors(
        session, conv, summary_upto_msg_id=5, layer1_from_msg_id=6
    )
    resp = client.get("/api/v1/conversations", params={"user_key": "u-side2"})
    assert resp.json()["conversations"][0]["summarized"] is True


async def test_conversation_messages_returns_original_text(client, session):
    from app.repositories import conversation_repo, message_repo

    conv = await conversation_repo.get_or_create(session, None, "u-msgs")
    await message_repo.add_message(session, conv.id, role="user", content="我要退货")
    await message_repo.add_message(session, conv.id, role="assistant", content="麻烦提供订单号")

    resp = client.get(
        f"/api/v1/conversations/{conv.id}/messages", params={"user_key": "u-msgs"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [m["content"] for m in body["messages"]] == ["我要退货", "麻烦提供订单号"]


async def test_conversation_messages_404_for_other_user(client, session):
    from app.repositories import conversation_repo

    conv = await conversation_repo.get_or_create(session, None, "u-owner")
    resp = client.get(f"/api/v1/conversations/{conv.id}/messages", params={"user_key": "u-other"})
    assert resp.status_code == 404


async def test_resume_endpoint_passes_decision_and_card(client, monkeypatch):
    """卡片上点按钮 → 决定与看到的内容一起递回图里;这条路径不传新消息。"""
    from sse_starlette.sse import ServerSentEvent

    seen: dict = {}

    async def fake_chat(session_id, message, conversation_id=None, **kwargs):
        seen.update(
            {"session_id": session_id, "message": message, "conversation_id": conversation_id, **kwargs}
        )
        yield ServerSentEvent(data={"finish_reason": "stop"}, event="done")

    monkeypatch.setattr("app.api.routes.chat_with_agent", fake_chat)
    resp = client.post(
        "/api/v1/chat/resume",
        json={
            "session_id": "t1",
            "conversation_id": 7,
            "decision": "approve",
            "ticket_type": "售后",
            "description": "鞋子开胶",
        },
    )
    assert resp.status_code == 200
    assert "finish_reason" in resp.text
    assert seen["message"] == ""
    assert seen["conversation_id"] == 7
    assert seen["resume"] == {"decision": "approve", "ticket_type": "售后", "description": "鞋子开胶"}


async def test_resume_endpoint_rejects_unknown_decision(client):
    resp = client.post(
        "/api/v1/chat/resume",
        json={"session_id": "t1", "conversation_id": 7, "decision": "maybe"},
    )
    assert resp.status_code == 422
