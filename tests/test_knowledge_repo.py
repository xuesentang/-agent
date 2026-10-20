"""knowledge_repo 单测(SQLite 内存库,ch03 双写状态机的 MySQL 侧)。"""

from app.models.knowledge_chunk import KnowledgeChunk
from app.repositories import knowledge_repo


def make_chunk(**overrides) -> KnowledgeChunk:
    defaults = dict(
        category="退货政策",
        questions="退货政策是什么",
        answer="签收后 7 天内可无理由退货。",
        content_type="policy",
    )
    defaults.update(overrides)
    return KnowledgeChunk(**defaults)


async def test_add_chunks_assigns_ids_and_pending_status(session):
    chunks = [make_chunk(), make_chunk(category="商品 FAQ")]
    out = await knowledge_repo.add_chunks(session, chunks)

    assert [c.id for c in out] == [1, 2]
    assert all(c.vectorize_status == "pending" for c in out)
    assert all(c.vector_id is None for c in out)


async def test_get_pending_returns_only_pending_in_id_order(session):
    a, b, c = await knowledge_repo.add_chunks(
        session, [make_chunk(), make_chunk(), make_chunk()]
    )
    await knowledge_repo.mark_done(session, b.id, "2")

    pending = await knowledge_repo.get_pending(session)
    assert [c.id for c in pending] == [a.id, c.id]


async def test_get_pending_respects_limit(session):
    await knowledge_repo.add_chunks(session, [make_chunk() for _ in range(5)])
    assert len(await knowledge_repo.get_pending(session, limit=2)) == 2


async def test_reset_vector_status_brings_done_chunks_back_to_pending(session):
    """删掉向量集合后如果不把状态退回 pending,这些块会被永久跳过。

    实踩:357 行的库重建集合后 Milvus 只有 336 条,少的 21 块不报错、也不在检索里。
    """
    a, b = await knowledge_repo.add_chunks(session, [make_chunk(), make_chunk()])
    await knowledge_repo.mark_done(session, a.id, "1")

    changed = await knowledge_repo.reset_vector_status(session)

    assert changed == 1
    pending = await knowledge_repo.get_pending(session)
    assert {c.id for c in pending} == {a.id, b.id}
    assert all(c.vector_id is None for c in pending)


async def test_reset_vector_status_keeps_pending_chunks_untouched(session):
    (a,) = await knowledge_repo.add_chunks(session, [make_chunk()])
    assert await knowledge_repo.reset_vector_status(session) == 0
    assert [c.id for c in await knowledge_repo.get_pending(session)] == [a.id]


async def test_mark_done_backfills_vector_id_and_status(session):
    (chunk,) = await knowledge_repo.add_chunks(session, [make_chunk()])
    await knowledge_repo.mark_done(session, chunk.id, "42")

    refreshed = await knowledge_repo.get_by_ids(session, [chunk.id])
    assert refreshed[0].vector_id == "42"
    assert refreshed[0].vectorize_status == "done"


async def test_mark_done_unknown_id_is_noop(session):
    await knowledge_repo.mark_done(session, 999, "x")  # 不抛异常


async def test_get_by_ids_empty_input(session):
    assert await knowledge_repo.get_by_ids(session, []) == []


async def test_get_by_ids_returns_in_id_order_not_query_order(session):
    a, b = await knowledge_repo.add_chunks(session, [make_chunk(), make_chunk()])
    out = await knowledge_repo.get_by_ids(session, [b.id, a.id])
    assert [c.id for c in out] == [a.id, b.id]


async def test_link_neighbors_chains_in_order(session):
    a, b, c = await knowledge_repo.add_chunks(
        session, [make_chunk(), make_chunk(), make_chunk()]
    )
    await knowledge_repo.link_neighbors(session, [[a, b, c]])

    rows = {r.id: r for r in await knowledge_repo.get_by_ids(session, [a.id, b.id, c.id])}
    assert rows[a.id].prev_chunk_id is None
    assert rows[a.id].next_chunk_id == b.id
    assert rows[b.id].prev_chunk_id == a.id
    assert rows[b.id].next_chunk_id == c.id
    assert rows[c.id].prev_chunk_id == b.id
    assert rows[c.id].next_chunk_id is None


async def test_link_neighbors_does_not_cross_document_boundary(session):
    """两篇文档各 2 块:每篇首块无 prev、末块无 next,第一篇末块也不指向第二篇首块。"""
    a, b, c, d = await knowledge_repo.add_chunks(session, [make_chunk() for _ in range(4)])
    await knowledge_repo.link_neighbors(session, [[a, b], [c, d]])

    rows = {r.id: r for r in await knowledge_repo.get_by_ids(session, [a.id, b.id, c.id, d.id])}
    assert rows[a.id].prev_chunk_id is None and rows[a.id].next_chunk_id == b.id
    assert rows[b.id].prev_chunk_id == a.id and rows[b.id].next_chunk_id is None
    assert rows[c.id].prev_chunk_id is None and rows[c.id].next_chunk_id == d.id
    assert rows[d.id].prev_chunk_id == c.id and rows[d.id].next_chunk_id is None


def test_normalize_question_strips_whitespace_punct_case():
    assert knowledge_repo.normalize_question(" 退货 政策是什么？ ") == "退货政策是什么"
    assert knowledge_repo.normalize_question("How do I return?") == "howdoireturn"
    assert knowledge_repo.normalize_question("") == ""


async def test_existing_questions_splits_multiline(session):
    await knowledge_repo.add_chunks(
        session, [make_chunk(questions="退货政策是什么\n怎么退货")]
    )
    got = await knowledge_repo.existing_questions(session)
    assert got == {"退货政策是什么", "怎么退货"}


def test_embedding_text_joins_three_fields():
    chunk = make_chunk(category=" 退货政策 ", questions=" 怎么退货 ", answer=" 7 天内可退 ")
    assert chunk.embedding_text() == "退货政策\n怎么退货\n7 天内可退"
