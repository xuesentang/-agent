from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.main import app
from app.services import chat as chat_service


class FakeModel:
    async def astream(self, messages):
        yield AIMessageChunk(content="你好呀")


def test_chat_stream_endpoint_returns_event_stream(monkeypatch):
    monkeypatch.setattr(chat_service, "get_chat_model", lambda: FakeModel())
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/stream", json={"session_id": "t1", "message": "你好"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in resp.text
    assert "event: done" in resp.text


def test_empty_message_rejected():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/stream", json={"session_id": "t1", "message": "   "}
    )
    assert resp.status_code == 422


def test_unknown_session_id_works(monkeypatch):
    monkeypatch.setattr(chat_service, "get_chat_model", lambda: FakeModel())
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/stream", json={"session_id": "brand-new", "message": "在吗"}
    )
    assert resp.status_code == 200
