"""Offline knowledge import, indexing and conversation extraction commands."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from customer_service.config import Settings, build_model
from customer_service.db import make_engine
from customer_service.knowledge_extract import LLMQAExtractor, extract_batches, promote_staged
from customer_service.knowledge_ingest import clean_retired, index_pending
from customer_service.knowledge_parser import parse_markdown, retire_stale_chunks
from customer_service.knowledge_store import KnowledgeStore, NewChunk, import_faq
from customer_service.models import Base, QAExtractionStaging, KnowledgeChunk
from customer_service.vector_store import BGEM3Embedder, MilvusVectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init-db", "import", "import-faq", "index", "extract", "promote"])
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("DATABASE_URL is required")
    engine = make_engine(url)
    if engine.dialect.name == "mysql" and engine.url.database != "customer_service":
        raise SystemExit("Knowledge database is restricted to customer_service")
    factory = sessionmaker(engine, expire_on_commit=False)
    store = KnowledgeStore(factory)
    if args.action == "init-db":
        Base.metadata.create_all(engine, tables=[KnowledgeChunk.__table__, QAExtractionStaging.__table__])
        print("Knowledge tables ready")
    elif args.action == "import":
        if not args.paths:
            raise SystemExit("import requires Markdown paths")
        chunks: list[NewChunk] = []
        for filename in args.paths:
            path = Path(filename)
            parsed = parse_markdown(path.read_text(encoding="utf-8"), path.as_posix())
            store.add_chunks(parsed)
            retire_stale_chunks(store, path.as_posix(), {chunk.source_key for chunk in parsed})
            chunks.extend(parsed)
        print(f"Imported {len(chunks)} chunks")
    elif args.action == "import-faq":
        print(f"Imported {import_faq(factory, store)} FAQ rows")
    elif args.action == "index":
        uri = os.getenv("MILVUS_URI", "").strip()
        token = os.getenv("MILVUS_TOKEN", "").strip()
        if not uri or not token:
            raise SystemExit("MILVUS_URI and MILVUS_TOKEN are required")
        vectors = MilvusVectors(uri, token)
        vectors.ensure_collection()
        while clean_retired(store, vectors):
            pass
        embedder = BGEM3Embedder()
        total = 0
        while True:
            count = index_pending(store, embedder, vectors)
            if not count:
                break
            total += count
        print(f"Indexed {total} chunks")
    elif args.action == "extract":
        settings = Settings.from_env()
        count = extract_batches(factory, LLMQAExtractor(build_model(settings), settings.structured_output_method), args.batch_size)
        print(f"Staged {count} QA pairs")
    elif args.action == "promote":
        print(f"Promoted {promote_staged(factory, store)} QA pairs")
    engine.dispose()


if __name__ == "__main__":
    main()
