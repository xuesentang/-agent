"""CH09 数据层：review_queue / eval_runs / intent_usage / 池表新列。"""

from __future__ import annotations

import pytest

from app.models.review_queue import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED
from app.repositories import eval_run_repo, low_confidence_repo, review_repo


# ---------------------------------------------------------------- review_queue


@pytest.mark.asyncio
async def test_create_starts_pending_with_count_one(session):
    row = await review_repo.create(
        session, normalized_question="如何申请退货？", ai_suggested_answer="在订单页点退货"
    )
    assert row.review_status == STATUS_PENDING
    assert row.occurrence_count == 1
    assert row.normalized_question == "如何申请退货？"


@pytest.mark.asyncio
async def test_bump_accumulates_without_new_rows(session):
    row = await review_repo.create(session, normalized_question="退货运费谁出？")
    await review_repo.bump(session, row.id)
    await review_repo.bump(session, row.id)
    again = await review_repo.get(session, row.id)
    assert again.occurrence_count == 3
    assert len(await review_repo.list_by_status(session, STATUS_PENDING)) == 1
    # 累加不覆盖标准化问题文本
    assert again.normalized_question == "退货运费谁出？"


@pytest.mark.asyncio
async def test_bump_unknown_id_returns_none(session):
    assert await review_repo.bump(session, 999) is None


@pytest.mark.asyncio
async def test_list_orders_by_occurrence_then_id(session):
    a = await review_repo.create(session, normalized_question="A")
    b = await review_repo.create(session, normalized_question="B")
    for _ in range(3):
        await review_repo.bump(session, b.id)
    rows = await review_repo.list_by_status(session, STATUS_PENDING)
    assert [r.normalized_question for r in rows] == ["B", "A"]
    assert a.occurrence_count == 1


@pytest.mark.asyncio
async def test_approve_writes_answer_and_status_together(session):
    row = await review_repo.create(session, normalized_question="Q")
    done = await review_repo.mark_approved(session, row.id, "  核准答案  ")
    assert done.review_status == STATUS_APPROVED
    assert done.approved_answer == "核准答案"  # 去空白
    counts = await review_repo.counts(session)
    assert counts == {STATUS_APPROVED: 1}


@pytest.mark.asyncio
async def test_reject_keeps_answer_empty(session):
    row = await review_repo.create(session, normalized_question="Q", ai_suggested_answer="示例")
    done = await review_repo.mark_rejected(session, row.id)
    assert done.review_status == STATUS_REJECTED
    assert done.approved_answer is None


@pytest.mark.asyncio
async def test_dedupe_pool_only_contains_pending(session):
    a = await review_repo.create(session, normalized_question="待审的")
    b = await review_repo.create(session, normalized_question="已通过的")
    await review_repo.mark_approved(session, b.id, "答案")
    pool = await review_repo.list_for_dedupe(session)
    assert [r.normalized_question for r in pool] == ["待审的"]


# ---------------------------------------------------------------- 池表新列


@pytest.mark.asyncio
async def test_pool_row_keeps_recall_snapshot(session):
    snapshot = [
        {"chunk_id": 7, "section_path": "售后/退货", "score": 0.42, "snippet": "退货需在 7 天内"}
    ]
    row = await low_confidence_repo.add_question(
        session,
        raw_question="这个能退吗",
        source="retrieval_low_conf",
        reason="confidence_low: 0.38",
        retrieved_chunks=snapshot,
    )
    assert row.retrieved_chunks == snapshot


@pytest.mark.asyncio
async def test_list_unmerged_and_mark(session):
    a = await low_confidence_repo.add_question(session, raw_question="Q1", source="self_check")
    b = await low_confidence_repo.add_question(session, raw_question="Q2", source="user_feedback")
    pending = await low_confidence_repo.list_unmerged(session)
    assert [r.id for r in pending] == [a.id, b.id]

    await low_confidence_repo.set_matched_review(session, a.id, 42)
    assert [r.id for r in await low_confidence_repo.list_unmerged(session)] == [b.id]
    assert await low_confidence_repo.set_matched_review(session, 12345, 42) is False


@pytest.mark.asyncio
async def test_list_by_review_returns_contributors_newest_first(session):
    first = await low_confidence_repo.add_question(session, raw_question="Q1", source="self_check")
    second = await low_confidence_repo.add_question(session, raw_question="Q2", source="user_feedback")
    for row in (first, second):
        await low_confidence_repo.set_matched_review(session, row.id, 9)
    rows = await low_confidence_repo.list_by_review(session, 9)
    assert [r.raw_question for r in rows] == ["Q2", "Q1"]


@pytest.mark.asyncio
async def test_add_question_rejects_unknown_source(session):
    with pytest.raises(ValueError):
        await low_confidence_repo.add_question(session, raw_question="Q", source="wechat")


# ---------------------------------------------------------------- eval_runs


@pytest.mark.asyncio
async def test_eval_runs_list_old_to_new(session):
    for mrr in (0.60, 0.66):
        await eval_run_repo.add_eval_run(
            session,
            triggered_by="手动",
            dataset_size=31,
            metrics={"mrr": mrr, "recall_at_5": 0.84},
        )
    rows = await eval_run_repo.list_eval_runs(session, limit=10)
    assert [r.metrics["mrr"] for r in rows] == [0.60, 0.66]  # 趋势图从左到右


@pytest.mark.asyncio
async def test_eval_run_rejects_unknown_trigger(session):
    with pytest.raises(ValueError):
        await eval_run_repo.add_eval_run(session, triggered_by="cron", dataset_size=1, metrics={})


# ---------------------------------------------------------------- intent_usage


@pytest.mark.asyncio
async def test_usage_by_intent_aggregates_and_sorts(session):
    await eval_run_repo.add_usage(
        session, intent="物流", route="知识库", prompt_tokens=100, completion_tokens=50
    )
    await eval_run_repo.add_usage(
        session, intent="物流", route="知识库", prompt_tokens=300, completion_tokens=150
    )
    await eval_run_repo.add_usage(
        session, intent="退款", route="退款", prompt_tokens=100, completion_tokens=100
    )
    rows = await eval_run_repo.usage_by_intent(session)
    assert [r["intent"] for r in rows] == ["物流", "退款"]  # 烧得多的在前
    top = rows[0]
    assert top["turns"] == 2 and top["total_tokens"] == 600
    assert top["avg_tokens"] == 300.0
    assert top["share"] == round(600 / 800 * 100, 1)
    assert top["sources"] == ["estimated"]


@pytest.mark.asyncio
async def test_usage_row_defaults_to_estimated(session):
    row = await eval_run_repo.add_usage(
        session, intent="闲聊", route=None, prompt_tokens=10, completion_tokens=5
    )
    assert row.token_source == "estimated"
    assert row.total_tokens == 15


@pytest.mark.asyncio
async def test_usage_rejects_unknown_source(session):
    with pytest.raises(ValueError):
        await eval_run_repo.add_usage(
            session, intent="物流", route=None, prompt_tokens=1, completion_tokens=1, token_source="guess"
        )
