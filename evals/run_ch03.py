"""Real chapter-three acceptance against configured MySQL, BGE-M3 and Milvus."""

import json
import os

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from customer_service.config import Settings
from customer_service.db import make_engine
from customer_service.knowledge_ingest import clean_retired, index_pending
from customer_service.knowledge_search import KnowledgeSearch
from customer_service.knowledge_store import KnowledgeStore, import_faq
from customer_service.models import KnowledgeChunk
from customer_service.knowledge_parser import parse_markdown
from pathlib import Path
from customer_service.vector_store import BGEM3Embedder, MilvusVectors


def main() -> int:
    load_dotenv()
    uri = os.getenv("MILVUS_URI", "").strip()
    token = os.getenv("MILVUS_TOKEN", "").strip()
    if not uri or not token:
        raise SystemExit("MILVUS_URI and MILVUS_TOKEN are required")
    engine = make_engine(Settings.from_env().database_url)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    vectors = MilvusVectors(uri, token)
    vectors.ensure_collection()
    embedder = BGEM3Embedder()
    import_faq(factory, store)
    policy = Path(__file__).resolve().parents[1] / "knowledge" / "shipping-policy.md"
    parsed = parse_markdown(policy.read_text(encoding="utf-8"), "knowledge/shipping-policy.md")
    store.add_chunks(parsed)
    store.retire_stale("knowledge/shipping-policy.md", {chunk.source_key for chunk in parsed})
    while clean_retired(store, vectors):
        pass
    pending = store.pending()
    if not pending:
        with factory.begin() as session:
            row = session.query(KnowledgeChunk).order_by(KnowledgeChunk.id).first()
            row.vectorize_status = "pending"
            row.vector_id = None
        pending = store.pending()
    target_id = pending[0].id
    before_hits = vectors.client.query(collection_name=vectors.collection, filter=f"id == {target_id}", output_fields=["id"])
    try:
        index_pending(store, embedder, vectors, after_upsert=lambda: (_ for _ in ()).throw(RuntimeError("intentional interruption")))
    except RuntimeError as exc:
        if str(exc) != "intentional interruption":
            raise
    assert any(row.id == target_id for row in store.pending())
    indexed = 0
    while count := index_pending(store, embedder, vectors):
        indexed += count
    assert store.get_done([target_id])[0].vector_id == str(target_id)
    after_hits = vectors.client.query(collection_name=vectors.collection, filter=f"id == {target_id}", output_fields=["id"])
    assert len(after_hits) == 1
    matches = KnowledgeSearch(store, embedder, vectors).search("邮费是多少")
    passed = any("运费" in row.question and ("99" in row.answer or "8 元" in row.answer) for row in matches)
    print(json.dumps({"resumed_pending": indexed, "vectors_before": len(before_hits), "vectors_after": len(after_hits), "matches": [vars(row) for row in matches], "pass": passed}, ensure_ascii=False))
    engine.dispose()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
