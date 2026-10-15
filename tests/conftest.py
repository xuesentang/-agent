"""Shared pytest fixtures (CH02).

SQLite :memory: 每个连接是独立库,StaticPool 固定单连接共享同一内存库 —
这是单测库跑 SQLite 的标准手法(spec D8),MySQL 真库留给集成验收。
engine / session / session_factory 三个 fixture 共享同一 engine:
工具内部自开短会话(session_factory),与测试 fixture session 看到同一份数据。
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.models import Base


@pytest.fixture
async def engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s


@pytest.fixture
def session_factory(engine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
