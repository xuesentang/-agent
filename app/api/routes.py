"""HTTP API routes (spec D4 protocol; CH02: tools on the same SSE entry)."""

from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services.extract import ExtractionError, extract_after_sales
from app.services.tool_chat import chat_with_tools

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # CH02: 不传则新建会话;done 事件回传 conversation_id,前端带它续聊。
    conversation_id: int | None = None

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
    session: AsyncSession = Depends(get_session),
) -> AsyncIterable[ServerSentEvent]:
    async for event in chat_with_tools(
        request.session_id, request.message, request.conversation_id, session=session
    ):
        yield event


@router.post("/chat/extract")
async def chat_extract(request: ExtractRequest) -> dict[str, Any]:
    try:
        return await extract_after_sales(request.text)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
