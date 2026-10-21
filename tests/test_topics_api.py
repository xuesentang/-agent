"""CH10 主题分布只读接口。

三个点必须钉住:①空库也得是 200 且 17 类齐全 ②`total/distinct` 不能混报(池子有重复)
③未知类目不 500。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import get_session
from app.main import app
from app.models.low_confidence_question import LowConfidenceQuestion
from app.ml.topic.taxonomy import LABELS
from app.repositories import topic_repo


@pytest.fixture
async def client(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


async def _seed(session) -> None:
    """造 3 行池子,但只有 2 个不同的问题——照真实池子（重复测试入池）的样子。"""
    texts = [
        "订单 1003 的双肩包能退吗",
        "订单 1003 的双肩包能退吗",
        "W40 有宽楦版本吗",
    ]
    session.add_all(
        [
            LowConfidenceQuestion(raw_question=text, source="chat")
            for text in texts
        ]
    )
    await session.commit()
    ids = [row.id for row in (await topic_repo.list_all(session))]
    await topic_repo.upsert_many(
        session, [(ids[0], ["退换货", "尺码"]), (ids[1], ["退换货"]), (ids[2], ["尺码"])]
    )


def test_distribution_empty_returns_all_seventeen(client):
    resp = client.get("/api/v1/topics/distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0 and data["empty"] is True
    assert data["hint"]  # 空库要告诉人下一步跑什么
    assert [item["label"] for item in data["labels"]] == list(LABELS)
    assert data["model"]["threshold"] > 0


@pytest.mark.anyio
async def test_distribution_counts_multi_label_and_reports_distinct(client, session):
    await _seed(session)
    data = client.get("/api/v1/topics/distribution").json()
    assert data["total"] == 3
    assert data["pool_total"] == 3
    assert data["distinct"] == 2  # 重复入池不能虚报样本量
    counts = {item["label"]: item["count"] for item in data["labels"]}
    assert counts["退换货"] == 2 and counts["尺码"] == 2
    assert counts["物流"] == 0  # 0 也要出现,否则条形图会变形
    assert data["empty"] is False and data["hint"] is None


@pytest.mark.anyio
async def test_latest_returns_joined_rows(client, session):
    await _seed(session)
    data = client.get("/api/v1/topics/latest?limit=2").json()
    assert data["count"] == 2
    assert set(data["items"][0]) == {"question_id", "text", "labels", "classified_at"}


@pytest.mark.anyio
async def test_by_label_filters_and_rejects_unknown(client, session):
    await _seed(session)
    data = client.get("/api/v1/topics/by-label?label=尺码").json()
    assert data["count"] == 2
    assert all("尺码" in item["labels"] for item in data["items"])

    bad = client.get("/api/v1/topics/by-label?label=配送时效")
    assert bad.status_code == 200
    assert bad.json()["count"] == 0 and "未知类目" in bad.json()["error"]


def test_admin_page_is_served(client):
    resp = client.get("/admin/topics/")
    assert resp.status_code == 200
    assert "主题分布" in resp.text
    assert "/admin/topics/topics.js" in resp.text
