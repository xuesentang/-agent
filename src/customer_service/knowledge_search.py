"""Dense retrieval with MySQL as the answer authority."""

from customer_service.knowledge_store import KnowledgeStore
from customer_service.repository import FAQRow


class KnowledgeSearch:
    def __init__(self, store: KnowledgeStore, embedder, vectors) -> None:
        self.store = store
        self.embedder = embedder
        self.vectors = vectors

    def search(self, question: str, limit: int = 3) -> list[FAQRow]:
        vector = self.embedder.encode([question])[0]
        ids = self.vectors.search(vector, limit=limit)
        return [FAQRow(row.questions, row.answer, row.category) for row in self.store.get_done(ids)]
