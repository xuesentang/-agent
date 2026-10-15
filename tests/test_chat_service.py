from langchain_core.messages import AIMessageChunk

from app.core.session import SessionStore
from app.services.chat import stream_chat_reply


class FakeModel:
    def __init__(self, chunks=None, error=None):
        self._chunks = chunks or []
        self._error = error

    async def astream(self, messages):
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


async def test_stream_yields_token_then_done():
    store = SessionStore()
    model = FakeModel([AIMessageChunk(content="您好"), AIMessageChunk(content="，请讲")])
    events = [e async for e in stream_chat_reply("s1", "你好", store=store, model=model)]
    assert events[0].event == "token"
    assert events[0].data["content"] == "您好"
    assert events[1].data["content"] == "，请讲"
    assert events[-1].event == "done"
    assert events[-1].data["session_id"] == "s1"
    assert events[-1].data["finish_reason"] == "stop"


async def test_stream_persists_history_after_done():
    store = SessionStore()
    model = FakeModel([AIMessageChunk(content="您好，请讲")])
    _ = [e async for e in stream_chat_reply("s1", "你好", store=store, model=model)]
    history = store.get_history("s1")
    assert len(history) == 2
    assert history[0].content == "你好"
    assert history[1].content == "您好，请讲"


async def test_upstream_error_emits_sse_error_event():
    store = SessionStore()
    model = FakeModel(error=RuntimeError("401 unauthorized: bad key " + "x" * 500))
    events = [e async for e in stream_chat_reply("s1", "你好", store=store, model=model)]
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "UPSTREAM_ERROR"
    assert len(events[-1].data["message"]) <= 200
    # 出错时不得把半截对话写进历史
    assert store.get_history("s1") == []


async def test_concurrent_same_session_serialized():
    """Review Important #2: same-session concurrency must not interleave history."""
    import asyncio

    fast_model_seen = {}

    class SlowModel:
        async def astream(self, messages):
            await asyncio.sleep(0.05)
            yield AIMessageChunk(content="first-reply")

    class FastModel:
        async def astream(self, messages):
            fast_model_seen["messages"] = list(messages)
            yield AIMessageChunk(content="second-reply")

    store = SessionStore()

    async def drain(gen):
        return [e async for e in gen]

    await asyncio.gather(
        drain(stream_chat_reply("s-cc", "msg-one", store=store, model=SlowModel())),
        drain(stream_chat_reply("s-cc", "msg-two", store=store, model=FastModel())),
    )

    # Second request's prompt must already contain round one (serialized,
    # not raced): its human messages are [msg-one (history), msg-two (input)].
    second_human = [m.content for m in fast_model_seen["messages"] if m.type == "human"]
    assert second_human == ["msg-one", "msg-two"]
    history = store.get_history("s-cc")
    assert [m.content for m in history] == [
        "msg-one", "first-reply", "msg-two", "second-reply",
    ]


async def test_stream_injects_system_prompt_first():
    captured = {}

    class CapModel:
        async def astream(self, messages):
            captured["messages"] = list(messages)
            yield AIMessageChunk(content="ok")

    store = SessionStore()
    _ = [e async for e in stream_chat_reply("s1", "你好", store=store, model=CapModel())]
    from langchain_core.messages import SystemMessage

    assert isinstance(captured["messages"][0], SystemMessage)
    assert "小智" in captured["messages"][0].content
