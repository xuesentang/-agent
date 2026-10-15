"""HTTP API routes (spec D4 protocol)."""

from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator

from app.services.chat import stream_chat_reply
from app.services.extract import ExtractionError, extract_after_sales

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


@router.post("/chat/stream", response_class=EventSourceResponse, response_model=None)
async def chat_stream(
    request: ChatRequest,
) -> AsyncIterable[ServerSentEvent]:
    async for event in stream_chat_reply(request.session_id, request.message):
        yield event


@router.post("/chat/extract")
async def chat_extract(request: ExtractRequest) -> dict[str, Any]:
    try:
        return await extract_after_sales(request.text)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
