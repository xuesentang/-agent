import json
from random import Random

import pytest
from sqlalchemy.orm import sessionmaker

from customer_service.business_tools import build_business_tools
from customer_service.db import make_engine
from customer_service.knowledge_search import KnowledgeSearch
from customer_service.knowledge_store import KnowledgeStore, NewChunk
from customer_service.models import Base
from customer_service.repository import Repository


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectors:
    def search(self, vector, limit=3):
        return self.ids


def test_semantic_faq_preserves_tool_contract(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'rag.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    knowledge = KnowledgeStore(factory)
    row = knowledge.add_chunks([NewChunk("配送", "运费说明", "订单满 99 元包邮。", "shipping")])[0]
    knowledge.mark_done(row.id, str(row.id))
    vectors = FakeVectors()
    vectors.ids = [row.id]
    search = KnowledgeSearch(knowledge, FakeEmbedder(), vectors)
    repo = Repository(factory)
    cid = repo.create_conversation()
    tool = next(x for x in build_business_tools(repo, cid, Random(1), search) if x.name == "query_faq")
    result = json.loads(tool.invoke({"keyword": "邮费是多少"}))
    assert result == {"found": True, "matches": [{"question": "运费说明", "answer": "订单满 99 元包邮。", "category": "配送"}]}
    vectors.ids = []
    assert json.loads(tool.invoke({"keyword": "无关问题"})) == {"found": False, "matches": []}


def test_vector_failure_is_not_reported_as_no_match(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'error.db'}")
    Base.metadata.create_all(engine)

    class BrokenVectors:
        def search(self, vector, limit=3):
            raise ConnectionError("vector unavailable")

    search = KnowledgeSearch(KnowledgeStore(sessionmaker(engine)), FakeEmbedder(), BrokenVectors())
    with pytest.raises(ConnectionError, match="vector unavailable"):
        search.search("邮费是多少")
