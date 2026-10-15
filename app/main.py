"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="电商智能客服", version="0.1.0")
app.include_router(router)
