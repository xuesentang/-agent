"""CH09 审核后台接口：队列、详情、通过写库、驳回、趋势、按意图 token。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import review as review_api
from app.db import get_session
from app.main import app
from app.rag.embeddings import FakeEmbedder
from app.rag.vectorstore import FakeMilvus, MilvusStore
from app.repositories import eval_run_repo, low_confidence_repo, review_repo


@pytest.fixture
async def client(engine, monkeypatch):
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with maker() as s:
            yield s

    # 通过审核要向量化：换成假嵌入 + 假向量库，别在单测里加载 2G 模型
    monkeypatch.setattr(
        review_api, "_index_assets", lambda: (FakeEmbedder(), MilvusStore(client=FakeMilvus()))
    )
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


async def _seed(session, question="退货运费由谁承担？", **kw):
    return await review_repo.create(session, normalized_question=question, **kw)


# ------------------------------------------------------------------ 队列


def test_empty_queue_returns_hint_not_404(client):
    resp = client.get("/api/v1/review/queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["empty"] is True
    assert body["items"] == []
    assert "run_flywheel" in body["hint"]


@pytest.mark.asyncio
async def test_queue_orders_by_occurrence(client, session):
    quiet = await _seed(session, "发票怎么开？")
    hot = await _seed(session, "退货运费由谁承担？")
    hot.occurrence_count = 9
    await session.commit()

    items = client.get("/api/v1/review/queue").json()["items"]

    assert [i["id"] for i in items] == [hot.id, quiet.id]
    assert items[0]["occurrence_count"] == 9
    assert items[0]["review_status"] == "待审"


@pytest.mark.asyncio
async def test_queue_filters_by_status(client, session):
    row = await _seed(session)
    await review_repo.mark_rejected(session, row.id)

    assert client.get("/api/v1/review/queue").json()["count"] == 0
    rejected = client.get("/api/v1/review/queue", params={"status": "驳回"}).json()
    assert rejected["count"] == 1


# ------------------------------------------------------------------ 详情


@pytest.mark.asyncio
async def test_detail_includes_raw_questions_and_snapshot(client, session):
    row = await _seed(session)
    await low_confidence_repo.add_question(
        session,
        raw_question="运费谁出啊",
        source="user_feedback",
        conversation_id=None,
        retrieved_chunks=[{"n": 1, "chunk_id": 5, "section_path": "退货政策", "score": 0.41, "snippet": "7 天内"}],
    )
    pool = (await low_confidence_repo.list_unmerged(session))[0]
    await low_confidence_repo.set_matched_review(session, pool.id, row.id)

    body = client.get(f"/api/v1/review/{row.id}").json()

    assert body["normalized_question"] == "退货运费由谁承担？"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["raw_question"] == "运费谁出啊"
    assert body["sources"][0]["snapshot"][0]["section_path"] == "退货政策"


def test_detail_404_for_unknown_id(client):
    assert client.get("/api/v1/review/999").status_code == 404


# ------------------------------------------------------------------ 通过 / 驳回


@pytest.mark.asyncio
async def test_approve_writes_knowledge_chunk(client, session):
    row = await _seed(session)
    resp = client.post(f"/api/v1/review/{row.id}/approve", json={"answer": "寄回运费由消费者承担。"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["created"] is True

    # 用接口回读，不再碰 ORM session：TestClient 跑在另一个事件循环里
    detail = client.get(f"/api/v1/review/{row.id}").json()
    assert detail["review_status"] == "通过"
    assert detail["approved_answer"] == "寄回运费由消费者承担。"


@pytest.mark.asyncio
async def test_approve_rejects_blank_answer(client, session):
    row = await _seed(session)
    resp = client.post(f"/api/v1/review/{row.id}/approve", json={"answer": ""})
    assert resp.status_code == 422  # 契约层就拦住，不进服务


def test_approve_404_for_unknown_id(client):
    resp = client.post("/api/v1/review/999/approve", json={"answer": "随便"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reject_changes_status_only(client, session):
    row = await _seed(session)
    resp = client.post(f"/api/v1/review/{row.id}/reject")
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "驳回"


def test_reject_404_for_unknown_id(client):
    assert client.post("/api/v1/review/999/reject").status_code == 404


# ------------------------------------------------------------------ 趋势


def test_trend_empty_returns_200(client):
    body = client.get("/api/v1/review/trend").json()
    assert body == {"count": 0, "items": []}


@pytest.mark.asyncio
async def test_trend_oldest_first(client, session):
    for i in range(3):
        await eval_run_repo.add_eval_run(
            session,
            triggered_by="手动",
            dataset_size=30,
            metrics={"strategy": "hybrid_rerank", "r5": 0.8, "r10": 0.9, "mrr": 0.7 + i / 100},
        )
    items = client.get("/api/v1/review/trend").json()["items"]
    assert [round(i["mrr"], 2) for i in items] == [0.7, 0.71, 0.72]
