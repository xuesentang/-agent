"""Adapters for BGE-M3 dense vectors and a Milvus collection."""


import os
from threading import Lock


class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = os.getenv("BGE_M3_MODEL_PATH", "").strip() or model_name
        self._model = None
        self._lock = Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.model_name, use_fp16=False, devices="cpu")
            result = self._model.encode(texts, batch_size=4, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vectors = result["dense_vecs"].tolist()
        if any(len(vector) != 1024 for vector in vectors):
            raise ValueError("BGE-M3 dense vectors must have 1024 dimensions")
        return vectors


class MilvusVectors:
    def __init__(self, uri: str, token: str, collection: str = "knowledge") -> None:
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token)
        self.collection = collection

    def ensure_collection(self) -> None:
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=1024,
                primary_field_name="id",
                id_type="int",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )

    def upsert(self, chunk_id: int, vector: list[float]) -> None:
        self.client.upsert(collection_name=self.collection, data={"id": chunk_id, "vector": vector})

    def search(self, vector: list[float], limit: int = 3) -> list[int]:
        hits = self.client.search(collection_name=self.collection, data=[vector], limit=limit, output_fields=[])
        return [int(hit["id"]) for hit in hits[0]]

    def delete(self, chunk_id: int) -> None:
        self.client.delete(collection_name=self.collection, ids=[chunk_id])
