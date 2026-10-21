"""CH09 按意图 token 估算：估算口径、落表、聚合、写失败不影响主链路。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import get_session
from app.main import app
from app.models.intent_usage import IntentUsage
from app.services import usage_service


# ------------------------------------------------------------------ 估算


def test_estimate_turn_counts_history_into_prompt():
    short = usage_service.estimate_turn({"resolved_query": "邮费多少", "reply": "包邮。"})
    long = usage_service.estimate_turn(
        {
            "resolved_query": "邮费多少",
            "history": [AIMessage(content="上一轮说了很多很多关于运费的话" * 10)],
            "reply": "包邮。",
        }
    )
    assert long[0] > short[0]
    assert long[1] == short[1]  # completion 只看答复


def test_estimate_turn_completion_tracks_reply_length():
    brief = usage_service.estimate_turn({"reply": "包邮。"})
    verbose = usage_service.estimate_turn({"reply": "包邮。" * 60})
    assert verbose[1] > brief[1]


def test_estimate_turn_never_negative():
    prompt, completion = usage_service.estimate_turn({})
    assert prompt >= 0 and completion >= 0


# ------------------------------------------------------------------ 落表


@pytest.mark.asyncio
async def test_record_turn_writes_estimated_row(session):
    await usage_service.record_turn(
        session,
        intent="商品咨询",
        route="knowledge",
        prompt_tokens=1200,
        completion_tokens=80,
        conversation_id=None,
        model="test-model",
    )
    row = (await session.execute(select(IntentUsage))).scalars().one()
    assert row.intent == "商品咨询"
    assert row.total_tokens == 1280
    assert row.token_source == "estimated"  # 口径必须留在行里


@pytest.mark.asyncio
async def test_record_turn_skips_when_intent_missing(session):
    await usage_service.record_turn(
        session, intent=None, route=None, prompt_tokens=10, completion_tokens=1
    )
    count = (await session.execute(select(func.count()).select_from(IntentUsage))).scalar()
    assert count == 0


@pytest.mark.asyncio
async def test_record_turn_swallows_db_errors(session, monkeypatch):
    """统计是旁路：写挂只记日志，不能把一轮对话搞崩。"""

    async def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(usage_service.eval_run_repo, "add_usage", boom)
    await usage_service.record_turn(
        session, intent="物流", route="business", prompt_tokens=1, completion_tokens=1
    )


# ------------------------------------------------------------------ 聚合


@pytest.mark.asyncio
async def test_by_intent_shares_and_average(session):
    for intent, p, c in (("商品咨询", 1000, 100), ("商品咨询", 800, 80), ("物流", 200, 20)):
        await usage_service.record_turn(
            session, intent=intent, route="x", prompt_tokens=p, completion_tokens=c
        )

    data = await usage_service.by_intent(session)

    assert data["total_tokens"] == 2200
    assert data["turns"] == 3
    first = data["items"][0]
    assert first["intent"] == "商品咨询"
    assert first["share"] == pytest.approx(90.0, abs=0.1)  # 1980 / 2200
    assert first["avg_tokens"] == pytest.approx(990.0, abs=0.1)
    assert data["items"][1]["intent"] == "物流"
    assert data["token_source"] == "estimated"


@pytest.mark.asyncio
async def test_by_intent_empty(session):
    data = await usage_service.by_intent(session)
    assert data["empty"] is True
    assert data["items"] == []
    assert data["total_tokens"] == 0


# ------------------------------------------------------------------ 接口


@pytest.fixture
def client(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_usage_endpoint_returns_200_when_empty(client):
    resp = client.get("/api/v1/usage/by-intent")
    assert resp.status_code == 200
    assert resp.json()["empty"] is True
    assert "估算" in resp.json()["note"]
