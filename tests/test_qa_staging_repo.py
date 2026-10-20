"""qa_staging_repo 单测(SQLite 内存库,ch03 挖知识暂存表)。"""

from app.repositories import qa_staging_repo


async def test_add_batch_writes_extracted_rows(session):
    rows = await qa_staging_repo.add_batch(
        session,
        batch_no="b1",
        pairs=[("怎么退货", "7 天内"), ("运费多少", "满 99 包邮")],
        source_ref="conv-1",
    )
    assert len(rows) == 2
    assert all(r.status == "extracted" for r in rows)
    assert all(r.batch_no == "b1" and r.source_ref == "conv-1" for r in rows)


async def test_list_by_status_filters(session):
    await qa_staging_repo.add_batch(session, "b1", [("q1", "a1")])
    await qa_staging_repo.add_batch(session, "b2", [("q2", "a2")])

    extracted = await qa_staging_repo.list_by_status(session, "extracted")
    assert len(extracted) == 2
    assert len(await qa_staging_repo.list_by_status(session, "extracted", batch_no="b2")) == 1
    assert await qa_staging_repo.list_by_status(session, "kept") == []


async def test_set_status_updates_single_row(session):
    (row,) = await qa_staging_repo.add_batch(session, "b1", [("q", "a")])
    await qa_staging_repo.set_status(session, row.id, "kept")

    kept = await qa_staging_repo.list_by_status(session, "kept")
    assert [r.id for r in kept] == [row.id]


async def test_set_status_bulk(session):
    rows = await qa_staging_repo.add_batch(
        session, "b1", [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
    )
    await qa_staging_repo.set_status_bulk(session, [rows[0].id, rows[2].id], "discarded")

    discarded = await qa_staging_repo.list_by_status(session, "discarded")
    assert sorted(r.id for r in discarded) == sorted([rows[0].id, rows[2].id])
    extracted = await qa_staging_repo.list_by_status(session, "extracted")
    assert [r.id for r in extracted] == [rows[1].id]


async def test_set_status_bulk_empty_is_noop(session):
    await qa_staging_repo.set_status_bulk(session, [], "kept")
