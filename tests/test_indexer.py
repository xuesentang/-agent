"""indexer 单测:三格映射 + 双写状态机 + 中断重跑幂等(SQLite + FakeMilvus + FakeEmbedder)。"""

import pytest

from app.models.knowledge_chunk import KnowledgeChunk
from app.rag.chunker import RawChunk
from app.rag.embeddings import FakeEmbedder
from app.rag.indexer import (
    BuildInterrupted,
    build_index,
    to_knowledge_chunks,
    _parent_path,
    _split_qa,
)
from app.rag.vectorstore import FakeMilvus, MilvusStore
from app.repositories import knowledge_repo

POLICY_MD = """# 退货政策

## 7 天无理由退货

本店支持签收后 7 天内无理由退货,商品需保持吊牌齐全、附件完整。"""

FAQ_MD = """# 商品常见问题

## 尺码与规格

Q: 你们跑鞋偏码吗?
A: 正常足弓建议按平时运动鞋尺码选购。

Q: 有宽楦版本吗?
A: 部分热门款有宽楦,详情页有标注。"""


# ---------------------------------------------------------------- 三格映射


def test_policy_mapping_questions_is_heading_category_is_parent():
    raws = [
        RawChunk("退货政策 > 7 天无理由退货", "7 天无理由退货", "本店支持退货。", False),
        RawChunk("退货政策", "退货政策", "总则。", False),
    ]
    chunks = to_knowledge_chunks(raws, "policy")
    assert chunks[0].questions == "7 天无理由退货"
    assert chunks[0].category == "退货政策"
    assert chunks[1].questions == "退货政策"  # 单级路径:category=questions=自身
    assert chunks[1].category == "退货政策"


def test_faq_mapping_extracts_real_questions():
    raws = [RawChunk("商品常见问题 > 尺码与规格", "尺码与规格", "Q: 你们跑鞋偏码吗?\nA: 按平时尺码选购。", False)]
    (chunk,) = to_knowledge_chunks(raws, "faq")
    assert chunk.questions == "你们跑鞋偏码吗?"
    assert chunk.answer == "按平时尺码选购。"
    assert chunk.content_type == "faq"


def test_faq_mapping_without_q_lines_falls_back_to_heading():
    raws = [RawChunk("商品常见问题 > 备注", "备注", "本店默认发顺丰。", False)]
    (chunk,) = to_knowledge_chunks(raws, "faq")
    assert chunk.questions == "备注"
    assert chunk.answer == "本店默认发顺丰。"


def test_parent_path_helper():
    assert _parent_path("A > B > C") == "A > B"
    assert _parent_path("A") == "A"
    assert _parent_path("") == ""


def test_split_qa_helper():
    q, a = _split_qa("Q: 问题一\nA: 答案一\nQ: 问题二\nA: 答案二")
    assert q == "问题一\n问题二"
    assert a == "答案一\n答案二"


# ---------------------------------------------------------------- 双写状态机


def _write_doc(tmp_path, name: str, md: str):
    p = tmp_path / name
    p.write_text(md, encoding="utf-8")
    return p


async def test_build_index_happy_path(session_factory, tmp_path):
    doc = _write_doc(tmp_path, "policy.md", POLICY_MD)
    store, embedder = MilvusStore(client=FakeMilvus()), FakeEmbedder()

    report = await build_index(
        session_factory=session_factory,
        store=store,
        embedder=embedder,
        docs=[(doc, "policy", "")],
    )
    assert report.chunks_inserted == 1
    assert report.chunks_vectorized == 1
    assert not report.interrupted

    async with session_factory() as session:
        assert await knowledge_repo.get_pending(session) == []  # 全部 done
        chunks = await knowledge_repo.get_by_ids(session, [1])
    assert chunks[0].vector_id == "1"
    assert len(store._client._rows) == 1


async def test_interrupt_then_rerun_completes_pending(session_factory, tmp_path):
    """验收标准 2 单测版:跑一批后中断 → 重跑 → pending 补齐且无重复向量。"""
    doc = _write_doc(tmp_path, "policy.md", POLICY_MD)
    store, embedder = MilvusStore(client=FakeMilvus()), FakeEmbedder()

    # 造 3 批以上的量:同一文档灌 5 份不同内容(内容键不同才会插入)
    docs = []
    for i in range(5):
        p = _write_doc(tmp_path, f"p{i}.md", POLICY_MD.replace("退货政策", f"退货政策{i}"))
        docs.append((p, "policy", ""))

    with pytest.raises(BuildInterrupted):
        await build_index(
            session_factory=session_factory,
            store=store,
            embedder=embedder,
            docs=docs,
            batch_size=1,
            stop_after_batches=2,
        )

    async with session_factory() as session:
        left = await knowledge_repo.get_pending(session)
    assert len(left) == 3  # 5 块 - 已向量化 2 块

    report = await build_index(
        session_factory=session_factory,
        store=store,
        embedder=embedder,
        docs=[],  # 重跑不传文档:纯补向量化模式
        batch_size=1,
    )
    assert report.chunks_vectorized == 3
    assert report.chunks_inserted == 0

    async with session_factory() as session:
        assert await knowledge_repo.get_pending(session) == []
    assert len(store._client._rows) == 5  # 主键对齐,无重复行


async def test_rerun_same_docs_no_duplicate_insert(session_factory, tmp_path):
    doc = _write_doc(tmp_path, "policy.md", POLICY_MD)
    kwargs = dict(session_factory=session_factory, store=MilvusStore(client=FakeMilvus()), embedder=FakeEmbedder(), docs=[(doc, "policy", "")])

    await build_index(**kwargs)
    report2 = await build_index(**kwargs)

    assert report2.chunks_inserted == 0
    assert report2.chunks_skipped_dupe == 1  # 内容键命中,跳过
    async with session_factory() as session:
        rows = await knowledge_repo.get_by_ids(session, [1])
    assert len(rows) == 1


async def test_links_do_not_cross_document_boundary(session_factory, tmp_path):
    """两篇文档各切 1 块:两块的 prev/next 都是 None,不被串成一条链。"""
    d1 = _write_doc(tmp_path, "a.md", POLICY_MD)
    d2 = _write_doc(tmp_path, "b.md", POLICY_MD.replace("退货政策", "换货政策"))
    store, embedder = MilvusStore(client=FakeMilvus()), FakeEmbedder()

    report = await build_index(
        session_factory=session_factory,
        store=store,
        embedder=embedder,
        docs=[(d1, "policy", ""), (d2, "policy", "")],
    )
    assert report.chunks_inserted == 2

    async with session_factory() as session:
        rows = await knowledge_repo.get_by_ids(session, [1, 2])
    assert rows[0].prev_chunk_id is None and rows[0].next_chunk_id is None
    assert rows[1].prev_chunk_id is None and rows[1].next_chunk_id is None


async def test_build_index_empty_docs_only_recovers_pending(session_factory):
    store, embedder = MilvusStore(client=FakeMilvus()), FakeEmbedder()
    report = await build_index(
        session_factory=session_factory, store=store, embedder=embedder, docs=[]
    )
    assert report.chunks_vectorized == 0
    assert report.docs == 0
