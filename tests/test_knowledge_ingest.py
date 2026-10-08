import pytest
from sqlalchemy.orm import sessionmaker

from customer_service.db import make_engine
from customer_service.knowledge_ingest import clean_retired, index_pending
from customer_service.knowledge_store import KnowledgeStore, NewChunk
from customer_service.models import Base


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.rows = {}

    def upsert(self, chunk_id, vector):
        self.rows[chunk_id] = vector

    def delete(self, chunk_id):
        self.rows.pop(chunk_id, None)


def test_retry_after_vector_write_uses_same_id(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'retry.db'}")
    Base.metadata.create_all(engine)
    store = KnowledgeStore(sessionmaker(engine, expire_on_commit=False))
    item = NewChunk("配送", "运费说明", "满 99 元包邮。", "policy#shipping")
    row = store.add_chunks([item])[0]
    vector = FakeVectorStore()
    with pytest.raises(RuntimeError, match="interrupted"):
        index_pending(store, FakeEmbedder(), vector, after_upsert=lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
    assert len(vector.rows) == 1
    assert store.pending()[0].id == row.id
    assert index_pending(store, FakeEmbedder(), vector) == 1
    assert len(vector.rows) == 1
    assert store.get_done([row.id])[0].vector_id == str(row.id)


def test_retired_chunk_is_deleted_from_vector_store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'retire.db'}")
    Base.metadata.create_all(engine)
    store = KnowledgeStore(sessionmaker(engine, expire_on_commit=False))
    row = store.add_chunks([NewChunk("政策", "旧运费", "旧说明。", "policy.md#old")])[0]
    vectors = FakeVectorStore()
    assert index_pending(store, FakeEmbedder(), vectors) == 1
    store.retire_stale("policy.md", set())
    assert store.get_done([row.id]) == []
    assert clean_retired(store, vectors) == 1
    assert vectors.rows == {}
    assert clean_retired(store, vectors) == 0
