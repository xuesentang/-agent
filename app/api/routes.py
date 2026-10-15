"""HTTP API routes (spec D4 protocol)."""

from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator

from app.services.chat import stream_chat_reply

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


@router.post("/chat/stream", response_class=EventSourceResponse, response_model=None)
async def chat_stream(
    request: ChatRequest,
) -> AsyncIterable[ServerSentEvent]:
    async for event in stream_chat_reply(request.session_id, request.message):
        yield event
