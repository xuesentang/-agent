"""embeddings 单测:接口契约 + FakeEmbedder 确定性(不加载真模型)。"""

from app.rag.embeddings import DIMENSION, FakeEmbedder


def test_fake_embedder_dimension():
    assert FakeEmbedder().dimension == 1024 == DIMENSION


def test_fake_embedder_same_text_same_vector():
    e = FakeEmbedder()
    assert e.embed_texts(["退货政策"]) == e.embed_texts(["退货政策"])


def test_fake_embedder_different_text_different_vector():
    e = FakeEmbedder()
    a, b = e.embed_texts(["退货政策", "运费规则"])
    assert a != b


def test_fake_embedder_preserves_batch_order():
    e = FakeEmbedder()
    out = e.embed_texts(["甲", "乙", "丙"])
    assert len(out) == 3
    assert out[0] == e.embed_texts(["甲"])[0]
    assert out[1] == e.embed_texts(["乙"])[0]


def test_fake_embedder_empty_batch():
    assert FakeEmbedder().embed_texts([]) == []
