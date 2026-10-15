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
        # 长连接断线自愈:MySQL wait_timeout 掐掉空闲连接后自动重连。
        pool_pre_ping=True,
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request."""
    async with get_sessionmaker()() as session:
        yield session
