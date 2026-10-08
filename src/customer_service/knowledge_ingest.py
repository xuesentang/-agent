"""Resume-safe pending knowledge vectorization."""

from collections.abc import Callable

from customer_service.knowledge_store import KnowledgeStore, vector_text


def index_pending(store: KnowledgeStore, embedder, vectors, limit: int = 100, after_upsert: Callable[[], None] | None = None) -> int:
    rows = store.pending(limit)
    if not rows:
        return 0
    embeddings = embedder.encode([vector_text(row) for row in rows])
    if len(embeddings) != len(rows):
        raise ValueError("Embedding count does not match chunk count")
    for row, embedding in zip(rows, embeddings):
        vectors.upsert(row.id, embedding)
        if after_upsert is not None:
            after_upsert()
        store.mark_done(row.id, str(row.id))
    return len(rows)


def clean_retired(store: KnowledgeStore, vectors, limit: int = 100) -> int:
    rows = store.retired(limit)
    for row in rows:
        vectors.delete(row.id)
        store.clear_retired(row.id)
    return len(rows)
