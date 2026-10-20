"""vectorstore 单测:FakeMilvus 替身测 MilvusStore 行为(不依赖 Docker/文件库)。"""

import pytest

from app.rag.vectorstore import COLLECTION_NAME, FakeMilvus, MilvusStore, VectorStoreError


def make_store() -> MilvusStore:
    return MilvusStore(client=FakeMilvus())


def make_vector(seed: int, dim: int = 1024) -> list[float]:
    return [float((seed + i) % 7) / 7.0 for i in range(dim)]


def test_ensure_collection_idempotent():
    store = make_store()
    store.ensure_collection()
    store.ensure_collection()
    # 第二次 has_collection 已为 True,不再重复建集合
    assert store._client.create_collection_calls == 1


def test_upsert_then_search_returns_sorted_topk():
    store = make_store()
    store.upsert(
        [
            {"chunk_id": 1, "embedding": make_vector(1)},
            {"chunk_id": 2, "embedding": make_vector(6)},
            {"chunk_id": 3, "embedding": make_vector(2)},
        ]
    )
    hits = store.search(make_vector(1), top_k=2)
    assert [h[0] for h in hits] == [1, 3]  # 与 chunk 1 同向的排最前
    assert hits[0][1] >= hits[1][1]


def test_upsert_same_pk_no_duplicate():
    store = make_store()
    store.upsert([{"chunk_id": 1, "embedding": make_vector(1)}])
    store.upsert([{"chunk_id": 1, "embedding": make_vector(2)}])
    hits = store.search(make_vector(2), top_k=5)
    assert len(hits) == 1  # 同主键覆盖,不重复行
    assert hits[0][0] == 1


def test_search_empty_collection_returns_empty():
    store = make_store()
    assert store.search(make_vector(1), top_k=5) == []


def test_upsert_empty_rows_is_noop():
    store = make_store()
    store.upsert([])
    assert store.search(make_vector(1)) == []


def test_upsert_error_wrapped_as_vector_store_error():
    class BoomClient(FakeMilvus):
        def upsert(self, rows):
            raise ConnectionError("milvus down")

    store = MilvusStore(client=BoomClient())
    with pytest.raises(VectorStoreError, match="upsert"):
        store.upsert([{"chunk_id": 1, "embedding": make_vector(1)}])


def test_search_error_wrapped_as_vector_store_error():
    class BoomClient(FakeMilvus):
        def search(self, vector, top_k):
            raise RuntimeError("collection not loaded")

    store = MilvusStore(client=BoomClient())
    with pytest.raises(VectorStoreError, match="search"):
        store.search(make_vector(1))


def test_ensure_collection_error_wrapped():
    class BoomClient(FakeMilvus):
        def has_collection(self, name):
            raise OSError("cannot reach milvus")

    store = MilvusStore(client=BoomClient())
    with pytest.raises(VectorStoreError, match="建集合"):
        store.ensure_collection()


def test_collection_name_constant():
    assert COLLECTION_NAME == "knowledge"


# ---------------------------------------------------------------- FLAT 索引(schema 显式建表)


def test_ensure_collection_creates_flat_index_with_explicit_schema():
    """索引类型定死 FLAT(faiss 窄字符 API 写不了中文路径,见 INDEX_TYPE 注释)。

    同时锁死显式 schema 建表路径:fast-path 的 dimension= 入口会把索引硬编码成
    AUTOINDEX,绕过它才能拿到 FLAT。
    """
    captured = {}

    class SpyClient(FakeMilvus):
        def create_collection(self, name, schema=None, index_params=None, **kwargs):
            captured["schema"] = schema
            captured["index_params"] = index_params
            super().create_collection(name, schema=schema, index_params=index_params, **kwargs)

    MilvusStore(client=SpyClient()).ensure_collection()
    schema, index_params = captured["schema"], captured["index_params"]
    field_names = {f.name for f in schema.fields}
    assert field_names == {"chunk_id", "embedding"}
    vec_field = next(f for f in schema.fields if f.name == "embedding")
    assert vec_field.params["dim"] == 1024
    assert len(index_params) == 1  # IndexParams 是 list-like
    param = index_params[0]
    assert param.field_name == "embedding"
    assert param.index_type == "FLAT"


def test_index_type_constant_is_flat():
    from app.rag.vectorstore import INDEX_TYPE

    assert INDEX_TYPE == "FLAT"


def test_search_retries_after_released_state():
    """跨进程撞 released 态:自动 load 重试一次,不炸检索。"""

    class ReleasedThenOkClient(FakeMilvus):
        def __init__(self):
            super().__init__()
            self.load_calls = 0
            self.search_calls = 0

        def search(self, collection_name, data, limit=10, output_fields=None, **kwargs):
            self.search_calls += 1
            if self.search_calls == 1:
                raise RuntimeError(
                    "Collection 'knowledge' is in state 'released'; call load() before search"
                )
            return super().search(collection_name, data, limit=limit, output_fields=output_fields, **kwargs)

        def load_collection(self, name, **kwargs):
            self.load_calls += 1

    store = MilvusStore(client=ReleasedThenOkClient())
    store.upsert([{"chunk_id": 1, "embedding": make_vector(1)}])
    hits = store.search(make_vector(1), top_k=1)
    assert hits[0][0] == 1
    assert store._client.load_calls == 1
    assert store._client.search_calls == 2


def test_search_non_released_error_still_raises():
    class BrokenClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise RuntimeError("some other failure")

    store = MilvusStore(client=BrokenClient())
    with pytest.raises(VectorStoreError, match="search"):
        store.search(make_vector(1))
