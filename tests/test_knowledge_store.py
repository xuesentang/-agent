from sqlalchemy.orm import sessionmaker

from customer_service.db import make_engine
from customer_service.knowledge_store import KnowledgeStore, NewChunk
from customer_service.models import Base
from customer_service.knowledge_store import import_faq
from customer_service.seed import init_database


def test_pending_import_is_repeat_safe_and_done_has_vector_id(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    Base.metadata.create_all(engine)
    store = KnowledgeStore(sessionmaker(engine, expire_on_commit=False))
    item = NewChunk("售后 / 配送", "运费说明", "订单满 99 元包邮。", "policy.md#shipping", "售后 / 配送 / 运费说明", "policy")
    first = store.add_chunks([item])[0]
    assert first.vectorize_status == "pending"
    assert first.vector_id is None
    assert store.add_chunks([item])[0].id == first.id
    assert [x.id for x in store.pending()] == [first.id]
    store.mark_done(first.id, str(first.id))
    assert store.pending() == []
    assert store.get_done([first.id])[0].answer == item.answer
    assert store.add_chunks([item])[0].id == first.id
    changed = NewChunk(item.category, item.questions, "订单满 88 元包邮。", item.source_key, item.section_path, item.content_type)
    revised = store.add_chunks([changed])[0]
    assert revised.id == first.id
    assert revised.vectorize_status == "pending"
    assert revised.vector_id is None
    assert revised.answer == changed.answer


def test_schema_has_two_new_tables():
    assert {"knowledge_chunks", "qa_extraction_staging"} <= set(Base.metadata.tables)


def test_import_existing_faq_is_idempotent(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'faq.db'}")
    init_database(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    assert import_faq(factory, store) == 2
    first_ids = [row.id for row in store.pending()]
    assert import_faq(factory, store) == 2
    assert [row.id for row in store.pending()] == first_ids
