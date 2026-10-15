"""routes TDD(CH02:SSE 入口挂工具,SQLite 覆盖 DB 依赖)。"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.tools.registry import ToolRegistry
from app.db import get_session
from app.main import app
from app.services import tool_chat as tool_chat_service


class FakeToolModel:
    def __init__(self, tool_calls=None):
        self._tool_calls = tool_calls or []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content="", tool_calls=self._tool_calls)

    async def astream(self, messages):
        yield AIMessageChunk(content="已为您查到：")


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


async def test_chat_stream_endpoint_returns_event_stream(client, monkeypatch):
    monkeypatch.setattr(tool_chat_service, "get_chat_model", lambda: FakeToolModel())
    resp = client.post("/api/v1/chat/stream", json={"session_id": "t1", "message": "你好"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in resp.text
    assert "event: done" in resp.text


async def test_empty_message_rejected(client):
    resp = client.post(
        "/api/v1/chat/stream", json={"session_id": "t1", "message": "   "}
    )
    assert resp.status_code == 422


async def test_tool_events_appear_in_stream(client, monkeypatch):
    reg = ToolRegistry()
    reg.register(fake_logistics)
    monkeypatch.setattr(
        tool_chat_service,
        "build_default_registry",
        lambda: reg,
    )
    monkeypatch.setattr(
        tool_chat_service,
        "get_chat_model",
        lambda: FakeToolModel(
            tool_calls=[{"name": "fake_logistics", "args": {"order_id": "1001"}, "id": "call_1", "type": "tool_call"}]
        ),
    )
    resp = client.post(
        "/api/v1/chat/stream",
        json={"session_id": "t1", "message": "订单1001的物流到哪了"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: tool_start" in body
    assert "fake_logistics" in body
    assert "event: tool_end" in body
    # 时序:tool_start 必须早于 token。
    assert body.index("event: tool_start") < body.index("event: token")
    assert body.index("event: tool_end") < body.index("event: done")
