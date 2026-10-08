import asyncio
import json

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from customer_service.chat import ChatService
from customer_service.db import make_engine
from customer_service.main import create_app
from customer_service.repository import Repository
from customer_service.models import FAQ
from customer_service.prompts import CHAT_PROMPT, FINAL_RESPONSE_RULE
from customer_service.repository import MessageRow
from customer_service.seed import init_database


class FakeDecision:
    def __init__(self, calls):
        self.calls = calls

    async def ainvoke(self, messages):
        return AIMessage(content="", tool_calls=self.calls)


class FakeModel:
    def __init__(self, calls):
        self.calls = calls
        self.final_messages = []

    def bind_tools(self, tools, **kwargs):
        assert kwargs["tool_choice"] == "auto"
        return FakeDecision(self.calls)

    async def astream(self, messages):
        self.final_messages.append(messages)
        yield AIMessageChunk(content="演示")
        yield AIMessageChunk(content="结果")


def test_tool_turn_sse_and_database(tmp_path):
    async def run():
        engine = make_engine(f"sqlite:///{tmp_path / 'chat.db'}")
        init_database(engine)
        repo = Repository(sessionmaker(engine, expire_on_commit=False))
        model = FakeModel([{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}])
        app = create_app(ChatService(model, repo, 4000), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/chat/stream", json={"message": "订单 1001 的物流到哪了"})
        blocks = [block.splitlines() for block in response.text.strip().split("\n\n")]
        names = [block[0][7:] for block in blocks]
        assert names == ["session", "tool_status", "tool_status", "token", "token", "done"]
        cid = json.loads(blocks[0][1][6:])["conversation_id"]
        assert [row.role for row in repo.list_messages(cid)] == ["user", "assistant", "tool", "assistant"]
        assert repo.list_messages(cid)[2].tool_call_id == "call_1"
        assert any(getattr(msg, "tool_call_id", None) == "call_1" for msg in model.final_messages[0])
    asyncio.run(run())


def test_same_conversation_requests_are_serialized(tmp_path):
    async def run():
        engine = make_engine(f"sqlite:///{tmp_path / 'parallel.db'}")
        init_database(engine)
        repo = Repository(sessionmaker(engine, expire_on_commit=False))
        cid = repo.create_conversation()

        class SlowModel(FakeModel):
            async def astream(self, messages):
                await asyncio.sleep(0.05)
                yield AIMessageChunk(content="答复")

        app = create_app(ChatService(SlowModel([]), repo, 4000), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await asyncio.gather(
                client.post("/chat/stream", json={"conversation_id": cid, "message": "第一问"}),
                client.post("/chat/stream", json={"conversation_id": cid, "message": "第二问"}),
            )
        assert [row.role for row in repo.list_messages(cid)] == ["user", "assistant", "user", "assistant"]
    asyncio.run(run())


def test_large_tool_result_stays_within_model_budget(tmp_path):
    async def run():
        engine = make_engine(f"sqlite:///{tmp_path / 'budget.db'}")
        init_database(engine)
        with sessionmaker(engine).begin() as session:
            faq = session.scalar(select(FAQ).where(FAQ.question == "退货政策"))
            faq.answer = "退货详情" * 5000
        repo = Repository(sessionmaker(engine, expire_on_commit=False))
        model = FakeModel([{"name": "query_faq", "args": {"keyword": "退货政策"}, "id": "call_1"}])
        app = create_app(ChatService(model, repo, 900), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/chat/stream", json={"message": "退货政策是什么"})
        assert response.status_code == 200
        assert count_tokens_approximately(model.final_messages[0]) <= 388
        cid = json.loads(response.text.split("\n\n")[0].splitlines()[1][6:])["conversation_id"]
        assert "退货详情" * 5000 in repo.list_messages(cid)[2].content
    asyncio.run(run())


def test_multiple_tool_calls_execute_none(tmp_path):
    async def run():
        engine = make_engine(f"sqlite:///{tmp_path / 'multiple.db'}")
        init_database(engine)
        repo = Repository(sessionmaker(engine, expire_on_commit=False))
        calls = [
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "a"},
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "b"},
        ]
        app = create_app(ChatService(FakeModel(calls), repo, 4000), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/chat/stream", json={"message": "订单和物流"})
        assert "event: error" in response.text
        cid = json.loads(response.text.split("\n\n")[0].splitlines()[1][6:])["conversation_id"]
        assert [row.role for row in repo.list_messages(cid)] == ["user"]
    asyncio.run(run())


def test_final_response_instruction_fits_token_budget(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'final_budget.db'}")
    init_database(engine)
    repo = Repository(sessionmaker(engine, expire_on_commit=False))
    cid = repo.create_conversation()
    user = "物流" * 100
    answer = "结果" * 100
    repo.append_messages(cid, [MessageRow("user", user), MessageRow("assistant", answer)])
    history = [HumanMessage(content=user), AIMessage(content=answer)]
    budget = count_tokens_approximately(CHAT_PROMPT.format_messages(history=history, message="继续")) + 512
    messages = ChatService(FakeModel([]), repo, budget).prepare(cid, "继续")
    final_messages = [SystemMessage(content=f"{messages[0].content}\n{FINAL_RESPONSE_RULE}"), *messages[1:]]
    assert count_tokens_approximately(final_messages) <= budget - 512
