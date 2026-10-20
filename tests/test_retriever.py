"""retriever 单测:embed→search→回表重排、空结果、脏主键跳过、异常上抛
(SQLite + FakeMilvus + FakeEmbedder)。降级策略在 query_faq 层,不在这层。"""

import pytest

from app.models.knowledge_chunk import KnowledgeChunk
from app.rag.embeddings import FakeEmbedder
from app.rag.retriever import Retriever, retrieve
from app.rag.vectorstore import FakeMilvus, MilvusStore, VectorStoreError
from app.repositories import knowledge_repo


async def _seed_chunks(session_factory, texts: list[str]) -> list[KnowledgeChunk]:
    """灌几条知识块(pending 无所谓,检索不挑状态),返回带 id 的实例。"""
    async with session_factory() as session:
        chunks = [
            KnowledgeChunk(category="测试", questions=f"问题{i}", answer=text)
            for i, text in enumerate(texts)
        ]
        await knowledge_repo.add_chunks(session, chunks)
        return chunks


def _fake_retriever(rows: list[dict] | None = None, embedder=None, client=None) -> Retriever:
    client = client or FakeMilvus()
    store = MilvusStore(client=client)
    if rows:
        store.upsert(rows)
    return Retriever(store=store, embedder=embedder or FakeEmbedder())


async def test_retrieve_returns_hits_in_score_order(session_factory):
    chunks = await _seed_chunks(session_factory, ["关于运费的说明", "关于退货的说明", "关于发票的说明"])
    embedder = FakeEmbedder()
    retriever = _fake_retriever(
        rows=[{"chunk_id": c.id, "embedding": embedder.embed_texts([c.embedding_text()])[0]} for c in chunks],
        embedder=embedder,
    )

    # query 精确等于目标块的向量化文本 → 余弦 1.0 必居首(FakeEmbedder 无语义,
    # 只有同文本才同向量);其余块分数由哈希定,只断言降序不断言相对次序。
    hits = await retrieve(
        session_factory=session_factory, query=chunks[1].embedding_text(), retriever=retriever
    )
    assert hits[0].chunk.id == chunks[1].id
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert len(hits) == 3


async def test_retrieve_empty_store_returns_empty(session_factory):
    retriever = _fake_retriever()
    assert await retrieve(session_factory=session_factory, query="任意问题", retriever=retriever) == []


async def test_retrieve_reorders_by_score_not_by_id(session_factory):
    """id 最大的块排第一——顺序以搜索分为准,不以回表 id 序为准。"""
    chunks = await _seed_chunks(session_factory, ["甲", "乙", "丙"])
    embedder = FakeEmbedder()
    retriever = _fake_retriever(
        rows=[{"chunk_id": c.id, "embedding": embedder.embed_texts([c.embedding_text()])[0]} for c in chunks],
        embedder=embedder,
    )

    hits = await retrieve(
        session_factory=session_factory, query=chunks[2].embedding_text(), retriever=retriever, top_k=1
    )
    assert len(hits) == 1
    assert hits[0].chunk.id == chunks[2].id  # 最大的 id 排第一


async def test_retrieve_top_k_limits_results(session_factory):
    chunks = await _seed_chunks(session_factory, ["甲", "乙", "丙"])
    embedder = FakeEmbedder()
    retriever = _fake_retriever(
        rows=[{"chunk_id": c.id, "embedding": embedder.embed_texts([c.embedding_text()])[0]} for c in chunks],
        embedder=embedder,
    )

    hits = await retrieve(
        session_factory=session_factory, query=chunks[0].embedding_text(), retriever=retriever, top_k=2
    )
    assert len(hits) == 2


async def test_retrieve_skips_ids_missing_in_mysql(session_factory):
    """Milvus 里有、MySQL 权威源没有的主键跳过,不撑翻整条检索。"""
    chunks = await _seed_chunks(session_factory, ["真实块一", "真实块二"])
    embedder = FakeEmbedder()
    retriever = _fake_retriever(
        rows=[
            {"chunk_id": c.id, "embedding": embedder.embed_texts([c.embedding_text()])[0]}
            for c in chunks
        ]
        + [{"chunk_id": 9999, "embedding": embedder.embed_texts(["幽灵块"])[0]}],
        embedder=embedder,
    )

    hits = await retrieve(session_factory=session_factory, query="真实块一", retriever=retriever)
    assert all(h.chunk.id != 9999 for h in hits)
    assert len(hits) >= 1


async def test_retrieve_embedder_error_propagates(session_factory):
    class BoomEmbedder:
        def embed_texts(self, texts):
            raise RuntimeError("model crashed")

    retriever = _fake_retriever(embedder=BoomEmbedder())
    with pytest.raises(RuntimeError, match="model crashed"):
        await retrieve(session_factory=session_factory, query="x", retriever=retriever)


async def test_retrieve_store_error_propagates(session_factory):
    class BoomClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise VectorStoreError("milvus down")

    retriever = _fake_retriever(client=BoomClient())
    with pytest.raises(VectorStoreError, match="milvus down"):
        await retrieve(session_factory=session_factory, query="x", retriever=retriever)
