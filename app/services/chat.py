"""Streaming customer-service chat service (spec D4 data flow)."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi.sse import ServerSentEvent
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import build_chat_prompt
from app.core.session import SessionStore

MAX_ERROR_MESSAGE_LENGTH = 200

# Process-wide shared history store. Restart loses it by design (spec §7).
_shared_store = SessionStore()
# Per-session lock: same-session requests are serialized so that
# read-history -> stream -> append-history cannot interleave (review #2).
_session_locks: dict[str, asyncio.Lock] = {}


def _get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


def _sanitize(message: str) -> str:
    """Never leak the API key through upstream error text (review Minor #9)."""
    key = get_settings().llm_api_key
    if key and key in message:
        message = message.replace(key, "***")
    return message[:MAX_ERROR_MESSAGE_LENGTH]


async def stream_chat_reply(
    session_id: str,
    message: str,
    store: SessionStore | None = None,
    model: ChatOpenAI | None = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield SSE events: token -> ... -> done, or a single error on failure."""
    store = store if store is not None else _shared_store
    model = model if model is not None else get_chat_model()

    async with _get_lock(session_id):
        try:
            history = store.trim_history(store.get_history(session_id))
            prompt_messages = build_chat_prompt().format_messages(
                messages=history, input=message
            )
        except Exception as exc:  # review Minor #6: pre-stream failures too
            yield ServerSentEvent(
                data={"code": "PROMPT_ERROR", "message": _sanitize(str(exc))},
                event="error",
            )
            return

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
                data={"code": "UPSTREAM_ERROR", "message": _sanitize(str(exc))},
                event="error",
            )
            return

        store.append_user(session_id, message)
        store.append_assistant(session_id, "".join(reply_parts))
        yield ServerSentEvent(
            data={"session_id": session_id, "finish_reason": finish_reason},
            event="done",
        )
