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
        def upsert(self, *args, **kwargs):
            raise ConnectionError("milvus down")

    store = MilvusStore(client=BoomClient())
    with pytest.raises(VectorStoreError, match="upsert"):
        store.upsert([{"chunk_id": 1, "embedding": make_vector(1)}])


def test_search_error_wrapped_as_vector_store_error():
    class BoomClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise RuntimeError("collection not loaded")

    store = MilvusStore(client=BoomClient())
    with pytest.raises(VectorStoreError, match="检索"):
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

    同时锁死两件事:
    - 显式 schema 建表:fast-path 的 dimension= 入口会把索引硬编码成 AUTOINDEX;
    - ch04 的字段集合:text(分词,产 BM25 稀疏向量)+ category(元数据过滤)+ sparse。
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
    assert field_names == {"chunk_id", "embedding", "text", "category", "sparse"}
    vec_field = next(f for f in schema.fields if f.name == "embedding")
    assert vec_field.params["dim"] == 1024
    # 两个索引:向量 FLAT + 稀疏 BM25
    by_field = {p.field_name: p for p in index_params}
    assert by_field["embedding"].index_type == "FLAT"
    assert by_field["sparse"].index_type == "SPARSE_INVERTED_INDEX"
    assert by_field["sparse"].to_dict()["metric_type"] == "BM25"
    # BM25 函数:text → sparse
    assert schema.functions[0].name == "bm25"


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

        def search(self, collection_name, data, anns_field="embedding", limit=10, output_fields=None, **kwargs):
            self.search_calls += 1
            if self.search_calls == 1:
                raise RuntimeError(
                    "Collection 'knowledge' is in state 'released'; call load() before search"
                )
            return super().search(
                collection_name,
                data,
                anns_field=anns_field,
                limit=limit,
                output_fields=output_fields,
                **kwargs,
            )

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
    with pytest.raises(VectorStoreError, match="检索"):
        store.search(make_vector(1))


# ---------------------------------------------------------------- ch04 关键词 / 混合 / 过滤


def _text_rows() -> list[dict]:
    return [
        {
            "chunk_id": 1,
            "embedding": make_vector(1),
            "text": "运费规则:满 99 元包邮,否则 8 元",
            "category": "售后政策",
        },
        {
            "chunk_id": 2,
            "embedding": make_vector(6),
            "text": "退货政策:签收 7 天内无理由退货",
            "category": "售后政策",
        },
        {
            "chunk_id": 3,
            "embedding": make_vector(3),
            "text": "发票说明:下单时填写抬头",
            "category": "开票",
        },
    ]


def test_search_sparse_hits_chinese_keyword():
    """BM25 那一路:Fake 用 2-gram 重叠打分,含「运费」的块必居首。"""
    store = make_store()
    store.upsert(_text_rows())
    hits = store.search_sparse("运费", top_k=5)
    assert hits and hits[0][0] == 1


def test_search_sparse_no_overlap_returns_empty():
    store = make_store()
    store.upsert(_text_rows())
    assert store.search_sparse("完全无关的词", top_k=5) == []


def test_hybrid_search_merges_dense_and_sparse_routes():
    """两路各召回一条:Fake 的本地 RRF 会把两路命中的 id 都留下。"""
    store = make_store()
    store.upsert(_text_rows())
    hits = store.hybrid_search(make_vector(1), "退货政策", top_k=5)
    ids = [h[0] for h in hits]
    assert 1 in ids and 2 in ids  # 1 来自向量相似、2 来自关键词命中
    assert ids[0] == 2  # 两路都靠前的排最前(RRF 是名次求和)


def test_category_filter_applies_before_search():
    """先过滤再检索:过滤掉 1、2 之后,关键词路只能命中发票块。"""
    store = make_store()
    store.upsert(_text_rows())
    hits = store.search_sparse("退货 运费 发票", top_k=5, expr='category == "开票"')
    assert [h[0] for h in hits] == [3]


def test_category_filter_expression_escapes_quotes():
    store = make_store()
    store.upsert(_text_rows())
    assert store.search_sparse("运费", top_k=5, expr='category == "售后政策"')[0][0] == 1
