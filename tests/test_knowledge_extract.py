from sqlalchemy.orm import sessionmaker

from customer_service.db import make_engine
from customer_service.knowledge_extract import stage_pairs, promote_staged
from customer_service.knowledge_extract import extract_batches
from customer_service.knowledge_extract import LLMQAExtractor
from customer_service.knowledge_store import KnowledgeStore
from customer_service.models import Base, QAExtractionStaging


def test_stage_then_global_dedup_and_repeat_safe_promotion(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'qa.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    stage_pairs(factory, "b1", "c1", [("运费是多少？", "满 99 元包邮。")])
    stage_pairs(factory, "b2", "c2", [(" 运费是多少? ", "满 99 元包邮。")])
    assert promote_staged(factory, store) == 1
    assert promote_staged(factory, store) == 0
    assert len(store.pending()) == 1
    assert store.pending()[0].source_key.startswith("qa:")
    with factory() as session:
        assert sorted(session.query(QAExtractionStaging.status).all()) == [("discarded",), ("kept",)]


def test_promotion_failure_leaves_staging_retryable(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'fail.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    stage_pairs(factory, "b1", "c1", [("退货条件？", "七天内。")])

    class BrokenStore:
        def add_chunks(self, chunks):
            raise RuntimeError("database unavailable")

    import pytest
    with pytest.raises(RuntimeError):
        promote_staged(factory, BrokenStore())
    with factory() as session:
        assert session.query(QAExtractionStaging).one().status == "extracted"


def test_extraction_batches_use_conversation_pairs(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'batch.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    from customer_service.repository import Repository, MessageRow
    repo = Repository(factory)
    for i in range(3):
        cid = repo.create_conversation()
        repo.append_messages(cid, [MessageRow("user", f"问题{i}"), MessageRow("assistant", f"答案{i}")])

    class FakeExtractor:
        def __init__(self):
            self.calls = []

        def extract(self, conversations):
            self.calls.append(conversations)
            return [(cid, f"问题{n}", f"答案{n}") for n, (cid, _) in enumerate(conversations)]

    extractor = FakeExtractor()
    assert extract_batches(factory, extractor, batch_size=2) == 3
    assert [len(batch) for batch in extractor.calls] == [2, 1]
    with factory() as session:
        assert session.query(QAExtractionStaging).count() == 3


def test_duplicate_across_batches_uses_one_stable_source(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'duplicate.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    stage_pairs(factory, "b1", "c1", [("运费是多少？", "满 99 元包邮。")])
    assert promote_staged(factory, store) == 1
    stage_pairs(factory, "b2", "c2", [("运费是多少?", "满 99 元包邮。")])
    assert promote_staged(factory, store) == 0
    assert len(store.pending()) == 1


def test_llm_extractor_keeps_conversation_reference():
    class Structured:
        def invoke(self, prompt):
            return {"pairs": [{"source_ref": "c1", "question": "如何退货？", "answer": "订单页申请退货。"}]}

    class Model:
        def with_structured_output(self, schema, method):
            assert method == "json_mode"
            return Structured()

    from customer_service.repository import MessageRow
    pairs = LLMQAExtractor(Model()).extract([("c1", [MessageRow("user", "如何退货？"), MessageRow("assistant", "订单页申请退货。")])])
    assert pairs == [("c1", "如何退货？", "订单页申请退货。")]


def test_two_distinct_pairs_from_same_conversation_survive(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'multi.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    stage_pairs(factory, "b1", "c1", [("退货怎么申请？", "在订单页申请。"), ("邮费是多少？", "满 99 元包邮。")])
    assert promote_staged(factory, store) == 2
    assert {row.questions for row in store.pending()} == {"退货怎么申请？", "邮费是多少？"}
