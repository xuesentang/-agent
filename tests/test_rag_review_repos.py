"""ch04 两张表的数据访问:低置信度问题池、编造个案台账(SQLite 内存库)。"""

import pytest

from app.repositories import faith_case_repo, low_confidence_repo


# ---------------------------------------------------------------- 低置信度池


async def test_add_question_persists_source_and_reason(session):
    row = await low_confidence_repo.add_question(
        session, raw_question="有没有货到付款", source="self_check", reason="证据讲的是运费"
    )
    assert row.id is not None
    assert row.source == "self_check"
    assert row.reason == "证据讲的是运费"
    assert row.created_at is not None


async def test_add_question_rejects_unknown_source(session):
    with pytest.raises(ValueError, match="未知入池入口"):
        await low_confidence_repo.add_question(session, raw_question="x", source="乱写")


async def test_list_by_source_filters(session):
    await low_confidence_repo.add_question(session, raw_question="a", source="self_check")
    await low_confidence_repo.add_question(session, raw_question="b", source="retrieval_low_conf")
    rows = await low_confidence_repo.list_by_source(session, "self_check")
    assert [r.raw_question for r in rows] == ["a"]
    assert await low_confidence_repo.count_all(session) == 2


# ---------------------------------------------------------------- 编造个案台账


def _case_kwargs(**over):
    base = dict(
        eval_id="A01",
        bucket="A_policy",
        query="退货运费谁出",
        answer="退货运费我们承担 [1]",
        reason="证据里没有这句",
        citations=[{"n": 1, "chunk_id": 3, "section_path": "运费规则"}],
        judge_model="glm-test",
    )
    base.update(over)
    return base


async def test_upsert_first_time_inserts_open_case(session):
    row, recurred = await faith_case_repo.upsert_case(session, **_case_kwargs())
    assert row.seen_count == 1
    assert row.status == "未解决"
    assert recurred is False


async def test_upsert_twice_updates_snapshot_and_counts(session):
    """一题一行:再判一次不新增行,只刷快照 + seen_count。"""
    await faith_case_repo.upsert_case(session, **_case_kwargs())
    row, recurred = await faith_case_repo.upsert_case(
        session, **_case_kwargs(answer="换了一版还是编的")
    )
    assert row.seen_count == 2
    assert row.answer == "换了一版还是编的"
    assert recurred is False  # 一直没修,不算复发

    rows = await faith_case_repo.list_open(session)
    assert len(rows) == 1


async def test_resolved_case_rejudged_reverts_to_open(session):
    """已解决的题又被判编造 = 复发:退回未解决,resolution 清空,resolved_at 保留。"""
    await faith_case_repo.upsert_case(session, **_case_kwargs())
    resolved = await faith_case_repo.resolve_case(session, "A01", "已解决", "补了「谁承担」一句")
    assert resolved is not None and resolved.status == "已解决"
    assert resolved.resolved_at is not None

    row, recurred = await faith_case_repo.upsert_case(session, **_case_kwargs())
    assert recurred is True
    assert row.status == "未解决"
    assert row.resolution is None
    assert row.resolved_at is not None  # 保留,用来标复发


async def test_resolve_requires_resolution_text(session):
    await faith_case_repo.upsert_case(session, **_case_kwargs())
    with pytest.raises(ValueError, match="处置说明不能为空"):
        await faith_case_repo.resolve_case(session, "A01", "已解决", "   ")


async def test_resolve_rejects_unknown_status(session):
    with pytest.raises(ValueError, match="非法处置状态"):
        await faith_case_repo.resolve_case(session, "A01", "搞定", "x")


async def test_resolve_unknown_case_returns_none(session):
    assert await faith_case_repo.resolve_case(session, "Z99", "已解决", "x") is None
