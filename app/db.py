"""Async engine & session factory (spec D1/D2)."""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """Process-wide engine. URL from settings (mysql+aiomysql in production,
    sqlite+aiosqlite in unit tests via monkeypatched settings)."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        # pool_pre_ping 在 aiomysql 上不可用:其 ping(reconnect) 签名要求参数,
        # SQLAlchemy 的 mysql 方言 do_ping 调 dbapi_connection.ping() 不带参 →
        # TypeError(集成验收抓到,单测 SQLite 无此路径)。改用 pool_recycle 规避:
        # 连接 1 小时回收,短于 MySQL 默认 wait_timeout(8h),不会拿到被掐断的死连接。
        pool_pre_ping=False,
        pool_recycle=3600,
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request."""
    async with get_sessionmaker()() as session:
        yield session
