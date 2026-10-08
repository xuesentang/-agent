import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from customer_service.chat import ChatService
from customer_service.config import Settings, build_model


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    message: NonBlank
    conversation_id: str | None = None
    user_id: str | None = None


class ExtractRequest(BaseModel):
    description: NonBlank


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(chat_service: ChatService, extraction_service) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=FileResponse)
    async def index():
        return Path(__file__).with_name("index.html")

    @app.post("/chat/stream")
    async def chat_stream(payload: ChatRequest):
        cid = payload.conversation_id
        if cid is None:
            cid = await asyncio.to_thread(chat_service.repository.create_conversation, payload.user_id or "guest")
        try:
            await asyncio.to_thread(chat_service.prepare, cid, payload.message)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async def generate() -> AsyncIterator[str]:
            yield _sse("session", {"conversation_id": cid})
            try:
                async for event in chat_service.stream(cid, payload.message):
                    yield _sse(event.name, event.data)
            except Exception:
                yield _sse("error", {"message": "Upstream generation failed"})
                return
            yield _sse("done", {})

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/aftersales/extract")
    async def extract(payload: ExtractRequest):
        if extraction_service is None:
            raise HTTPException(status_code=503, detail="Extraction service unavailable")
        try:
            return await extraction_service.extract(payload.description)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Structured extraction failed") from exc

    return app


def _build_app() -> FastAPI:
    from customer_service.aftersales import ExtractionService

    settings = Settings.from_env()
    model = build_model(settings)
    from sqlalchemy.orm import sessionmaker
    from customer_service.db import make_engine
    from customer_service.repository import Repository

    repository = Repository(sessionmaker(make_engine(settings.database_url), expire_on_commit=False))
    return create_app(
        ChatService(model, repository, settings.history_token_budget),
        ExtractionService(model, settings.structured_output_method),
    )
