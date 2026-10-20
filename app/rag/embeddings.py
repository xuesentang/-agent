"""嵌入适配器(ch03 spec D2/D3)。

接口定死:`embed_texts(texts) -> list[list[float]]`,1024 维(BGE-M3)。
当前实现 = 本地 sentence-transformers 加载 BGE-M3(D2 用户拍板「本地推理」,
CPU 推理);若日后切 OpenAI 兼容 /embeddings API,只需新增一个同接口实现,
上层(indexer/retriever)不动。

测试替身 FakeEmbedder:确定性 1024 维,不加载真模型、不联网。
"""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache

# BGE-M3 输出维度(定死;集合 schema 与适配器共用这个常量)
DIMENSION = 1024


class FakeEmbedder:
    """测试替身:同文本 → 同向量(sha256 播种随机),不同文本 → 不同向量。"""

    dimension = DIMENSION

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(seed)
            out.append([rng.uniform(-1.0, 1.0) for _ in range(self.dimension)])
        return out


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """进程内单例:模型加载一次(~2G 权重),后续 encode 直接复用。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LocalBGEEmbedder:
    """本地 BGE-M3 推理(sentence-transformers)。

    normalize_embeddings=True:向量归一化后配 Milvus COSINE 度量,
    点积即余弦相似度(BGE 官方推荐用法)。
    """

    dimension = DIMENSION

    def __init__(self, model_name: str | None = None) -> None:
        from app.config import get_settings

        self.model_name = model_name or get_settings().embedding_model
        self._model = _load_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]
