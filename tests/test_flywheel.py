"""CH09 入池与快照（需求 3 的前半），含 👍/👎 接口。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import get_session
from app.main import app
from app.repositories import low_confidence_repo
from app.rag.citations import Citation
from app.services import flywheel


def _citation(n: int, score: float, answer: str = "退货需在签收后 7 天内申请，超时只能换货") -> Citation:
    return Citation(
        n=n,
        chunk_id=100 + n,
        section_path="售后政策/退货",
        question="怎么退货",
        answer=answer,
        score=score,
    )


# ---------------------------------------------------------------- 快照


def test_snapshot_from_citations_keeps_five_and_trims_snippet():
    citations = [_citation(i, 0.9 - i * 0.05, "答" * 400) for i in range(1, 9)]
    snap = flywheel.snapshot_from_citations(citations)
    assert len(snap) == flywheel.SNAPSHOT_LIMIT
    assert set(snap[0]) == {"n", "chunk_id", "section_path", "score", "snippet"}
    assert len(snap[0]["snippet"]) == flywheel.SNIPPET_CHARS
    assert snap[0]["snippet"].endswith("…")
    assert snap[0]["score"] == 0.85  # 0.9 - 1*0.05


def test_snapshot_accepts_frontend_dicts():
    rows = [{"n": 1, "chunk_id": 3, "section_path": "A/B", "score": 0.5, "answer": "原文"}]
    snap = flywheel.snapshot_from_citations(rows)
    assert snap == [
        {"n": 1, "chunk_id": 3, "section_path": "A/B", "score": 0.5, "snippet": "原文"}
    ]


def test_sanitize_snapshot_rejects_junk_and_clamps_score():
    raw = [
        "not-a-dict",
        {"chunk_id": 1, "score": "0.9"},
        {"chunk_id": 2, "score": 5},  # 超出 [0,1] 要被夹住
    ]
    snap = flywheel.sanitize_snapshot(raw)
    assert len(snap) == 2
    assert snap[1]["score"] == 1.0  # 夹到上界
    assert snap[0]["score"] == 0.9  # 字符串数字转成 float


def test_sanitize_snapshot_returns_none_for_empty_or_wrong_type():
    assert flywheel.sanitize_snapshot(None) is None
    assert flywheel.sanitize_snapshot([]) is None
    assert flywheel.sanitize_snapshot("nope") is None
    assert flywheel.sanitize_snapshot(["junk"]) is None


# ---------------------------------------------------------------- 入池


@pytest.mark.asyncio
async def test_enqueue_stores_snapshot_from_citations(session):
    row = await flywheel.enqueue(
        session,
        raw_question="  这个能退吗  ",
        source=flywheel.SOURCE_RETRIEVAL,
        reason="confidence_low: 0.31",
        citations=[_citation(1, 0.41)],
    )
    assert row.raw_question == "这个能退吗"  # 空白压平
    assert row.retrieved_chunks[0]["chunk_id"] == 101
    assert row.source == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_enqueue_without_citations_leaves_snapshot_null(session):
    row = await flywheel.enqueue(
        session, raw_question="发票怎么开", source=flywheel.SOURCE_USER_FEEDBACK
    )
    assert row.retrieved_chunks is None


@pytest.mark.asyncio
async def test_enqueue_ignores_blank_question(session):
    assert await flywheel.enqueue(session, raw_question="   ", source=flywheel.SOURCE_SELF_CHECK) is None
    assert await low_confidence_repo.count_all(session) == 0


# ---------------------------------------------------------------- 👎 接口


@pytest.fixture
async def client(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_feedback_down_pools_question_with_snapshot(client):
    resp = client.post(
        "/api/v1/feedback",
        json={
            "conversation_id": None,
            "question": "我的订单能用花呗分期吗",
            "rating": "down",
            "citations": [
                {"n": 1, "chunk_id": 5, "section_path": "支付/方式", "score": 0.33, "answer": "支持微信与支付宝"}
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["pooled"] is True
    assert body["snapshot_size"] == 1


def test_feedback_up_does_not_pool(client):
    resp = client.post(
        "/api/v1/feedback",
        json={"question": "邮费多少", "rating": "up", "citations": []},
    )
    assert resp.status_code == 200
    assert resp.json()["pooled"] is False


def test_feedback_rejects_blank_question(client):
    resp = client.post("/api/v1/feedback", json={"question": "   ", "rating": "down"})
    assert resp.status_code == 422


def test_feedback_tolerates_missing_conversation(client):
    resp = client.post(
        "/api/v1/feedback",
        json={"conversation_id": 999999, "question": "能退吗", "rating": "down"},
    )
    assert resp.status_code == 200
    assert resp.json()["pooled"] is True
