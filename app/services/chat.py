"""Streaming customer-service chat service (spec D4 data flow)."""

from collections.abc import AsyncGenerator
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_openai import ChatOpenAI

from app.core.llm import get_chat_model
from app.core.prompts import build_chat_prompt
from app.core.session import SessionStore

MAX_ERROR_MESSAGE_LENGTH = 200

# Process-wide shared history store. Restart loses it by design (spec §7).
_shared_store = SessionStore()


async def stream_chat_reply(
    session_id: str,
    message: str,
    store: SessionStore | None = None,
    model: ChatOpenAI | None = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield SSE events: token -> ... -> done, or a single error on upstream failure."""
    store = store if store is not None else _shared_store
    model = model if model is not None else get_chat_model()

    history = store.trim_history(store.get_history(session_id))
    prompt_messages = build_chat_prompt().format_messages(messages=history, input=message)

    finish_reason = "stop"
    reply_parts: list[str] = []
    try:
        async for chunk in model.astream(prompt_messages):
            text = chunk.text
            if not text:
                # GLM-5.3-Flash thinking chunks carry no content; skip them (spec D6).
                continue
            reply_parts.append(text)
            chunk_finish = (chunk.response_metadata or {}).get("finish_reason")
            if chunk_finish:
                finish_reason = chunk_finish
            yield ServerSentEvent(data={"content": text}, event="token")
    except Exception as exc:
        yield ServerSentEvent(
            data={
                "code": "UPSTREAM_ERROR",
                "message": str(exc)[:MAX_ERROR_MESSAGE_LENGTH],
            },
            event="error",
        )
        return

    store.append_user(session_id, message)
    store.append_assistant(session_id, "".join(reply_parts))
    yield ServerSentEvent(
        data={"session_id": session_id, "finish_reason": finish_reason},
        event="done",
    )
