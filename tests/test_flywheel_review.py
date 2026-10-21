"""CH09 审核动作：通过走 ch03 双写、驳回不写库、二次审核改答案不新增块。"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.models.knowledge_chunk import KnowledgeChunk
from app.models.review_queue import STATUS_APPROVED, STATUS_REJECTED, ReviewQueue
from app.rag.embeddings import FakeEmbedder
from app.rag.vectorstore import FakeMilvus, MilvusStore
from app.repositories import knowledge_repo, review_repo
from app.services import flywheel

ANSWER = "签收后 7 天内可申请无理由退货，寄回运费由消费者承担。"


@pytest.fixture
def store() -> MilvusStore:
    return MilvusStore(client=FakeMilvus())


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


async def _chunks(session) -> list[KnowledgeChunk]:
    return list((await session.execute(select(KnowledgeChunk))).scalars().all())


# ------------------------------------------------------------------ 通过


@pytest.mark.asyncio
async def test_approve_writes_chunk_and_marks_done(session, store, embedder):
    row = await review_repo.create(session, normalized_question="退货运费由谁承担？")

    result = await flywheel.approve(session, row.id, ANSWER, store=store, embedder=embedder)

    assert result["created"] is True
    assert result["vectorized"] == 1
    chunks = await _chunks(session)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.category == flywheel.FLYWHEEL_CATEGORY
    assert chunk.questions == "退货运费由谁承担？"
    assert chunk.answer == ANSWER
    assert chunk.vectorize_status == "done"  # 双写走完才算数
    assert chunk.vector_id == str(chunk.id)

    refreshed = await review_repo.get(session, row.id)
    assert refreshed.review_status == STATUS_APPROVED
    assert refreshed.approved_answer == ANSWER


@pytest.mark.asyncio
async def test_approve_pushes_vector_into_store(session, store, embedder):
    row = await review_repo.create(session, normalized_question="发票怎么开？")
    await flywheel.approve(session, row.id, "下单时在备注里填抬头即可。", store=store, embedder=embedder)
    hits = store.search(embedder.embed_texts(["发票怎么开？"])[0], top_k=1)
    assert hits and hits[0][0] == 1  # (chunk_id, score)


@pytest.mark.asyncio
async def test_second_approve_updates_the_same_chunk(session, store, embedder):
    """同一个缺口改答案：更新块而不是再插一行，否则检索会出两条近似答案。"""
    row = await review_repo.create(session, normalized_question="退货运费由谁承担？")
    await flywheel.approve(session, row.id, ANSWER, store=store, embedder=embedder)

    result = await flywheel.approve(
        session, row.id, "签收后 7 天内可申请退货，运费由我们承担。", store=store, embedder=embedder
    )

    assert result["created"] is False
    chunks = await _chunks(session)
    assert len(chunks) == 1
    assert "由我们承担" in chunks[0].answer
    assert chunks[0].vectorize_status == "done"


@pytest.mark.asyncio
async def test_approve_requires_a_non_empty_answer(session, store, embedder):
    row = await review_repo.create(session, normalized_question="退货要理由吗？")
    with pytest.raises(ValueError):
        await flywheel.approve(session, row.id, "   ", store=store, embedder=embedder)
    assert await _chunks(session) == []
    assert (await review_repo.get(session, row.id)).review_status != STATUS_APPROVED


@pytest.mark.asyncio
async def test_approve_unknown_review_returns_none(session, store, embedder):
    assert await flywheel.approve(session, 999, ANSWER, store=store, embedder=embedder) is None
    assert await _chunks(session) == []


@pytest.mark.asyncio
async def test_approve_also_flushes_leftover_pending_chunks(session, store, embedder):
    """顺带补掉别的 pending：双写状态机本来就该把欠的向量化一起还上。"""
    await knowledge_repo.add_chunks(
        session, [KnowledgeChunk(category="退货政策", questions="能退吗", answer="7 天内可退")]
    )
    row = await review_repo.create(session, normalized_question="能开发票吗？")

    result = await flywheel.approve(session, row.id, "可以开电子发票。", store=store, embedder=embedder)

    assert result["vectorized"] == 2
    statuses = {c.vectorize_status for c in await _chunks(session)}
    assert statuses == {"done"}


# ------------------------------------------------------------------ 驳回


@pytest.mark.asyncio
async def test_reject_only_changes_status(session, store, embedder):
    row = await review_repo.create(session, normalized_question="你们老板是谁")
    out = await flywheel.reject(session, row.id)
    assert out.review_status == STATUS_REJECTED
    assert await _chunks(session) == []


@pytest.mark.asyncio
async def test_reject_unknown_review_returns_none(session):
    assert await flywheel.reject(session, 999) is None


@pytest.mark.asyncio
async def test_rejected_rows_leave_the_dedupe_pool(session):
    """驳回后再来同样的问题会新建一条，而不是并回被驳回的那条。"""
    row = await review_repo.create(session, normalized_question="你们老板是谁")
    await flywheel.reject(session, row.id)

    candidates = await review_repo.list_for_dedupe(session)
    assert candidates == []  # 只比待审的
    plan = await flywheel.dedupe(session, "你们老板是谁", model=None)
    assert plan["via"] == flywheel.VIA_NEW

    total = (await session.execute(select(func.count()).select_from(ReviewQueue))).scalar()
    assert total == 2
