import asyncio
import json

import httpx
from langchain_core.messages import AIMessageChunk
from sqlalchemy.orm import sessionmaker

from customer_service.chat import ChatService
from customer_service.db import make_engine
from customer_service.main import create_app
from customer_service.repository import Repository
from customer_service.seed import init_database


class FakeModel:
    def __init__(self):
        self.seen = []
        self.fail = False

    async def astream(self, messages):
        self.seen.append(messages)
        yield AIMessageChunk(content="你好")
        if self.fail:
            raise RuntimeError("upstream failed")
        yield AIMessageChunk(content="，请问")

    def bind_tools(self, tools, **kwargs):
        class Decision:
            async def ainvoke(self, messages):
                from langchain_core.messages import AIMessage
                return AIMessage(content="", tool_calls=[])
        return Decision()


def make_repo(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'stream.db'}")
    init_database(engine)
    return Repository(sessionmaker(engine, expire_on_commit=False))


def events(body):
    result = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        result.append((lines[0][7:], json.loads(lines[1][6:])))
    return result


def test_stream_and_second_turn(tmp_path):
    async def run():
        model = FakeModel()
        app = create_app(ChatService(model, make_repo(tmp_path), 4000), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/chat/stream", json={"message": "我的编号是 A"})
            first_events = events(first.text)
            assert [name for name, _ in first_events] == ["session", "token", "token", "done"]
            cid = first_events[0][1]["conversation_id"]
            second = await client.post("/chat/stream", json={"conversation_id": cid, "message": "我的编号是什么"})
            assert events(second.text)[0][1]["conversation_id"] == cid
            assert any(message.content == "我的编号是 A" for message in model.seen[1])
            assert any(message.content == "你好，请问" for message in model.seen[1])
            assert (await client.post("/chat/stream", json={"message": " "})).status_code == 422
            assert (await client.post("/chat/stream", json={"conversation_id": "bad", "message": "hi"})).status_code == 404
    asyncio.run(run())


def test_failed_stream_does_not_commit_turn(tmp_path):
    async def run():
        model = FakeModel()
        model.fail = True
        store = make_repo(tmp_path)
        app = create_app(ChatService(model, store, 4000), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/chat/stream", json={"message": "hello"})
        parsed = events(response.text)
        assert parsed[-1][0] == "error"
        assert [m.role for m in store.list_messages(parsed[0][1]["conversation_id"])] == ["user"]
    asyncio.run(run())
