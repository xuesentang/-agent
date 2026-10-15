"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(title="电商智能客服", version="0.2.0")
app.include_router(router)
# CH02: 聊天页(原生 HTML/JS,Vibe Coding 交付)。mount 在路由之后注册,
# /api/v1 更具体先匹配,互不覆盖。
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
