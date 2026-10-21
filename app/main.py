"""FastAPI application entrypoint."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.logging_setup import configure_logging

configure_logging()

logger = logging.getLogger("kefu_agent")


async def _mine_qa_loop() -> None:
    """对话挖知识后台循环:MINE_QA_ENABLED=true 时按 interval 跑一轮。

    单轮失败只记日志不退出循环(定时任务的命比一次成败长)。
    """
    from app.db import get_sessionmaker
    from app.rag.miner import mine_qa_pairs

    settings = get_settings()
    interval = settings.mine_qa_interval_hours * 3600
    while True:
        await asyncio.sleep(interval)
        try:
            report = await mine_qa_pairs(
                session_factory=get_sessionmaker(),
                batch_size=settings.mine_qa_batch_size,
            )
            logger.info("mine_qa done: %s", report.summary())
        except Exception:  # noqa: BLE001 - 定时任务单轮失败不拖垮服务
            logger.exception("mine_qa round failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    # CH07:上下文预算自检。装不下就该在启动时喊出来,而不是等用户聊到第 N 轮报 400
    try:
        from app.core.token_budget import log_startup_check

        log_startup_check()
    except Exception:  # noqa: BLE001 - 自检失败不阻塞启动
        logger.exception("context budget check failed")
    # 预热 RAG 检索器:嵌入模型 + 向量库集合在首个请求前就位,
    # 否则第一条 query_faq 要在工具超时内背下模型加载,必超时。
    try:
        from app.rag.retriever import warmup

        logger.info("retriever warmup: loading embedding model & vector store...")
        await asyncio.to_thread(warmup)
        logger.info("retriever warmup done")
    except Exception:  # noqa: BLE001 - 预热失败不阻塞启动,首条查询现加载
        logger.exception("retriever warmup failed; first query loads on demand")
    task = None
    if settings.mine_qa_enabled:
        task = asyncio.create_task(_mine_qa_loop())
        logger.info(
            "mine_qa scheduler started: every %dh, batch %d",
            settings.mine_qa_interval_hours,
            settings.mine_qa_batch_size,
        )
    yield
    if task is not None:
        task.cancel()


app = FastAPI(title="电商智能客服", version="0.3.0", lifespan=lifespan)
app.include_router(router)
# CH02: 聊天页(原生 HTML/JS,Vibe Coding 交付)。mount 在路由之后注册,
# /api/v1 更具体先匹配,互不覆盖。
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
