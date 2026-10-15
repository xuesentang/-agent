"""chat_with_tools 服务层 TDD(spec D4 单轮架构 / Review Focus 1/2/5/6)。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.core.tools.registry import ToolRegistry
from app.models.message import Message
from app.services.tool_chat import chat_with_tools


class FakeToolModel:
    """bind_tools().ainvoke() 定工具;astream() 流式收敛。second_tool_calls
    模拟 GLM 偶发二次仍要调工具的行为(触发兜底)。"""

    def __init__(
        self,
        tool_calls=None,
        reply_chunks=None,
        second_tool_calls=False,
        error=None,
        content="",
    ):
        self._tool_calls = tool_calls or []
        self._chunks = reply_chunks or [AIMessageChunk(content="好的")]
        self._second_tool_calls = second_tool_calls
        self._error = error
        self._content = content
        self.seen_messages = None

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.seen_messages = list(messages)
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content, tool_calls=self._tool_calls)

    async def astream(self, messages):
        if self._second_tool_calls:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_faq", "args": '{"keyword": "x"}', "id": "call_2", "index": 0}
                ],
            )
            return
        for chunk in self._chunks:
            yield chunk


@tool
def fake_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(fake_logistics)
    return reg


def _tool_call(name="fake_logistics", args=None, cid="call_1") -> dict:
    return {"name": name, "args": args or {"order_id": "1001"}, "id": cid, "type": "tool_call"}


async def _events(model, session, **kw):
    return [
        e
        async for e in chat_with_tools(
            "s1", "订单1001的物流到哪了", session=session, model=model,
            registry=kw.pop("registry", _registry()), **kw,
        )
    ]


async def test_no_tool_call_streams_tokens_then_done(session):
    events = await _events(FakeToolModel(), session)
    assert [e.event for e in events] == ["token", "done"]
    assert events[-1].data["session_id"] == "s1"
    assert events[-1].data["finish_reason"] == "stop"
    assert isinstance(events[-1].data["conversation_id"], int)


async def test_tool_call_emits_start_end_then_answer(session):
    model = FakeToolModel(tool_calls=[_tool_call()])
    events = await _events(model, session)
    kinds = [e.event for e in events]
    assert kinds == ["tool_start", "tool_end", "token", "done"]
    assert events[0].data["name"] == "fake_logistics"
    assert events[0].data["tool_call_id"] == "call_1"
    assert events[1].data["ok"] is True
    assert "北京转运中心" in events[1].data["summary"]


async def test_tool_trail_persisted_in_order(session):
    """Review Focus 2:user/assistant(tool_calls)/tool/assistant 四条对号落库。"""
    model = FakeToolModel(tool_calls=[_tool_call()])
    events = await _events(model, session)
    conv_id = events[-1].data["conversation_id"]

    from app.repositories import message_repo

    msgs = await message_repo.list_by_conversation(session, conv_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1].tool_calls[0]["id"] == "call_1"
    assert msgs[2].tool_call_id == "call_1"  # 对号入座
    assert "北京转运中心" in (msgs[2].content or "")


async def test_tool_failure_feeds_error_back_and_stream_continues(session):
    @tool
    def broken_tool(x: str) -> str:
        """坏工具。x: 任意"""
        raise RuntimeError("boom")

    reg = ToolRegistry()
    reg.register(broken_tool)
    model = FakeToolModel(
        tool_calls=[_tool_call(name="broken_tool", args={"x": "a"})],
    )
    events = await _events(model, session, registry=reg)
    assert events[1].event == "tool_end"
    assert events[1].data["ok"] is False
    # 流不挂起:错误已回灌,终答照常流出。
    assert events[-1].event == "done"
    assert any(e.event == "token" for e in events)


async def test_second_round_tool_calls_trigger_fallback(session):
    """spec §3:二次仍要工具 → 兜底摘要,不再执行。"""
    model = FakeToolModel(
        tool_calls=[_tool_call()],
        second_tool_calls=True,
    )
    events = await _events(model, session)
    assert events[-1].event == "done"
    fallback_tokens = [e for e in events if e.event == "token"]
    assert len(fallback_tokens) == 1
    assert "北京转运中心" in fallback_tokens[0].data["content"]


async def test_unknown_conversation_id_creates_new(session):
    """Review Focus 1:未知 conversation_id 静默新建,done 回传新 id。"""
    model = FakeToolModel()
    events = [
        e
        async for e in chat_with_tools(
            "s1", "在吗", conversation_id=99999, session=session, model=model, registry=_registry()
        )
    ]
    conv_id = events[-1].data["conversation_id"]
    assert conv_id != 99999


async def test_upstream_error_emits_error_and_persists_nothing(session):
    model = FakeToolModel(error=RuntimeError("401 unauthorized: bad key " + "x" * 500))
    events = await _events(model, session)
    assert [e.event for e in events] == ["error"]
    assert events[0].data["code"] == "UPSTREAM_ERROR"
    assert len(events[0].data["message"]) <= 200
    # CH01 语义:出错不落半截。
    from app.repositories import message_repo

    assert await message_repo.list_by_conversation(session, 999999) == []


async def test_concurrent_same_session_serialized(session):
    """同会话并发串行化(CH01 review #2 延续):第二轮的 prompt 已含第一轮历史。"""
    seen: dict = {}

    class SlowModel(FakeToolModel):
        async def astream(self, messages):
            await asyncio.sleep(0.05)
            yield AIMessageChunk(content="first-reply")

    class FastModel(FakeToolModel):
        async def astream(self, messages):
            seen["messages"] = list(messages)
            yield AIMessageChunk(content="second-reply")

    async def drain(model):
        return [e async for e in chat_with_tools(
            "s-cc", "msg-two" if model.__class__ is FastModel else "msg-one",
            session=session, model=model, registry=_registry(),
        )]

    await asyncio.gather(drain(SlowModel()), drain(FastModel()))

    from langchain_core.messages import HumanMessage

    second_human = [m.content for m in seen["messages"] if isinstance(m, HumanMessage)]
    assert second_human == ["msg-one", "msg-two"]


async def test_system_prompt_first_with_tool_guidance(session):
    model = FakeToolModel()
    await _events(model, session)
    from langchain_core.messages import SystemMessage

    assert isinstance(model.seen_messages[0], SystemMessage)
    assert "小智" in model.seen_messages[0].content
    assert "query_order" in model.seen_messages[0].content  # 工具指引在 system prompt 里
