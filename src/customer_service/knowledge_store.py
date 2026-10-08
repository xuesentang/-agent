"""MySQL authority for knowledge text and vectorization state."""

from dataclasses import dataclass
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from customer_service.models import FAQ, KnowledgeChunk


@dataclass(frozen=True)
class NewChunk:
    category: str
    questions: str
    answer: str
    source_key: str
    section_path: str | None = None
    content_type: str | None = None
    is_key_clause: bool = False


def vector_text(chunk: NewChunk | KnowledgeChunk) -> str:
    return f"分类：{chunk.category}\n问题：{chunk.questions}\n答案：{chunk.answer}"


def fingerprint(chunk: NewChunk) -> str:
    return sha256(vector_text(chunk).encode("utf-8")).hexdigest()


class KnowledgeStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def add_chunks(self, chunks: list[NewChunk]) -> list[KnowledgeChunk]:
        result = []
        with self.session_factory.begin() as session:
            for chunk in chunks:
                digest = fingerprint(chunk)
                row = session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.source_key == chunk.source_key))
                if row is None:
                    row = KnowledgeChunk(**vars(chunk), content_hash=digest)
                    session.add(row)
                    session.flush()
                elif row.content_hash != digest:
                    row.category = chunk.category
                    row.questions = chunk.questions
                    row.answer = chunk.answer
                    row.section_path = chunk.section_path
                    row.content_type = chunk.content_type
                    row.is_key_clause = chunk.is_key_clause
                    row.content_hash = digest
                    row.vector_id = None
                    row.vectorize_status = "pending"
                    session.flush()
                result.append(row)
            for row in result:
                session.expunge(row)
        return result

    def pending(self, limit: int = 100) -> list[KnowledgeChunk]:
        with self.session_factory() as session:
            rows = session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == "pending").order_by(KnowledgeChunk.id).limit(limit)).all()
            for row in rows:
                session.expunge(row)
            return rows

    def retired(self, limit: int = 100) -> list[KnowledgeChunk]:
        with self.session_factory() as session:
            rows = session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == "retired").order_by(KnowledgeChunk.id).limit(limit)).all()
            for row in rows:
                session.expunge(row)
            return rows

    def clear_retired(self, chunk_id: int) -> None:
        with self.session_factory.begin() as session:
            row = session.get(KnowledgeChunk, chunk_id)
            if row is not None and row.vectorize_status == "retired":
                row.vectorize_status = "retired_clean"

    def mark_done(self, chunk_id: int, vector_id: str) -> None:
        with self.session_factory.begin() as session:
            row = session.get(KnowledgeChunk, chunk_id)
            if row is None:
                raise KeyError(chunk_id)
            row.vector_id = vector_id
            row.vectorize_status = "done"

    def get_done(self, ids: list[int]) -> list[KnowledgeChunk]:
        if not ids:
            return []
        with self.session_factory() as session:
            rows = session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids), KnowledgeChunk.vectorize_status == "done")).all()
            for row in rows:
                session.expunge(row)
            by_id = {row.id: row for row in rows}
            return [by_id[i] for i in ids if i in by_id]

    def retire_stale(self, source: str, active_keys: set[str]) -> list[int]:
        with self.session_factory.begin() as session:
            escaped = source.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.source_key.like(f"{escaped}#%", escape="\\"))).all()
            retired = []
            for row in rows:
                if row.source_key not in active_keys and row.vectorize_status != "retired":
                    row.vectorize_status = "retired"
                    row.vector_id = None
                    retired.append(row.id)
            return retired


def import_faq(session_factory: sessionmaker, store: KnowledgeStore) -> int:
    with session_factory() as session:
        rows = session.scalars(select(FAQ).order_by(FAQ.id)).all()
        chunks = [NewChunk(row.category, row.question, row.answer, f"faq:{row.id}", row.question, "faq") for row in rows]
    store.add_chunks(chunks)
    return len(chunks)
