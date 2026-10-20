"""Milvus 封装(ch03 spec D3/D5)。

集合 knowledge 两个字段(fast-path 创建):
- chunk_id  Int64 主键,与 MySQL knowledge_chunks.id 严格对齐(双写幂等键);
- embedding FloatVector 1024 维(BGE-M3),COSINE 度量,FLAT 暴力索引(见 INDEX_TYPE 注释)。

原文(category/questions/answer)只存 MySQL,Milvus 只存向量——检索命中后回
MySQL 取原文(MySQL 是权威源);这样集合 schema 最小、无冗余。

upsert 语义:同主键重复写不报错、不产生重复行(建库重跑的前提)。
search 返回 [(chunk_id, score)] 按分降序;集合为空返回 []。
连接/查询异常统一包成 VectorStoreError,上层(retriever)据此降级 LIKE。

环境变量名注意事项:pymilvus 自己会读 `.env` 里的 `MILVUS_URI`(import 时解析,
相对路径直接抛 ConnectionConfigException),故本项目的配置项叫
`VECTOR_STORE_URI`,避开与 pymilvus 内部配置撞名。

测试替身 FakeMilvus:内存 dict + 余弦暴力扫,不依赖 Docker/文件库。
"""

from __future__ import annotations

import math

from app.rag.embeddings import DIMENSION

COLLECTION_NAME = "knowledge"

# 索引类型定死 FLAT(暴力精确检索),不用默认的 AUTOINDEX(HNSW 近似):
# 1. 演示量级(几百块)暴力全扫比近似检索更准,构建开销可忽略;
# 2. milvus-lite 的 FLAT 走 BruteForceIndex,索引持久化用 Python open()——
#    AUTOINDEX 走 faiss 的 write_index(const char*),Windows 上 faiss 用窄字符
#    ANSI API,UTF-8 编码的中文路径(D:\客服agent\...)直接 ENOENT,第二进程
#    load_collection 必崩(实踩:跨进程重开 Lite 文件库复现)。
INDEX_TYPE = "FLAT"


class VectorStoreError(RuntimeError):
    """Milvus 不可用(连接失败 / 集合不存在 / 查询异常)。"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class FakeMilvus:
    """内存替身,严格镜像 MilvusClient 高层接口(has_collection/create_collection/
    upsert/search),search 用余弦暴力扫;不依赖 Docker/文件库。

    与真客户端唯一差别:无索引/无持久化——单测只验调用契约与去重/排序语义。
    """

    def __init__(self) -> None:
        self._rows: dict[int, list[float]] = {}
        self._collections: set[str] = set()
        self.create_collection_calls = 0

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def load_collection(self, name: str, **kwargs) -> None:
        pass  # Fake 无加载态概念

    def create_collection(self, name, schema=None, index_params=None, **kwargs) -> None:
        self.create_collection_calls += 1
        self._collections.add(name)

    def upsert(self, collection_name: str, data: list[dict], **kwargs) -> dict:
        for r in data:
            self._rows[r["chunk_id"]] = r["embedding"]
        return {"upsert_count": len(data)}

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict]]:
        query = data[0]
        scored = [(cid, _cosine(query, vec)) for cid, vec in self._rows.items()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [[{"chunk_id": cid, "distance": score} for cid, score in scored[:limit]]]


class MilvusStore:
    """MilvusClient 薄封装;client 可注入(FakeMilvus 单测用),否则按 uri 真连。"""

    def __init__(self, uri: str | None = None, client=None) -> None:
        self._client = client
        if self._client is None:
            from app.config import get_settings

            from pymilvus import MilvusClient

            try:
                self._client = MilvusClient(uri or get_settings().vector_store_uri)
            except Exception as exc:  # noqa: BLE001 - 统一包成 VectorStoreError
                raise VectorStoreError(f"Milvus 连接失败:{exc}") from exc

    # ------------------------------------------------------------ 集合管理

    def ensure_collection(self) -> None:
        """幂等建集合(Lite 文件库 / 服务端同行为);已存在则跳过并确保已加载。

        显式 schema:主键 chunk_id(INT64)+ 向量 embedding(FLOAT_VECTOR 1024,COSINE,
        FLAT 暴力索引)。不建 category 标量字段——原文只存 MySQL(spec D3)。

        load_collection 不可省:Milvus Lite 文件库跨进程重开时集合是 released 态,
        不 load 直接 search 报 code=101。
        """
        try:
            if self._client.has_collection(COLLECTION_NAME):
                self._client.load_collection(COLLECTION_NAME)
                return
            # 显式 schema 建表(fast-path 的 dimension= 入口把索引硬编码成 AUTOINDEX,
            # 绕不开):chunk_id 主键 + embedding 向量两个字段,FLAT 暴力索引。
            from pymilvus import CollectionSchema, DataType, FieldSchema
            from pymilvus.milvus_client.index import IndexParams

            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                ]
            )
            index_params = IndexParams()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type="COSINE",
            )
            self._client.create_collection(
                COLLECTION_NAME, schema=schema, index_params=index_params
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus 建集合失败:{exc}") from exc

    # ------------------------------------------------------------ 写入 / 检索

    def drop_collection(self) -> None:
        """删集合重建用(演示 clean rebuild;旧 AUTOINDEX 集合不删会在 load 时崩)。"""
        try:
            if self._client.has_collection(COLLECTION_NAME):
                self._client.drop_collection(COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus 删集合失败:{exc}") from exc

    def upsert(self, rows: list[dict]) -> None:
        """rows: [{"chunk_id": int, "embedding": [float]*1024}]。"""
        if not rows:
            return
        try:
            self._client.upsert(COLLECTION_NAME, data=rows)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus upsert 失败:{exc}") from exc

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(chunk_id, score)] 按分降序;空集合 / 无命中返回 []。

        跨进程场景:建库进程 upsert 后,应用进程里已持有的 store 再 search 会撞
        「collection released」——先 load 再重试一次,不让一次状态不同步炸掉检索。
        """
        try:
            res = self._client.search(
                COLLECTION_NAME,
                data=[vector],
                limit=top_k,
            )
        except Exception as first_exc:  # noqa: BLE001
            if not _is_released_error(first_exc):
                raise VectorStoreError(f"Milvus search 失败:{first_exc}") from first_exc
            try:
                self._client.load_collection(COLLECTION_NAME)
                res = self._client.search(
                    COLLECTION_NAME,
                    data=[vector],
                    limit=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Milvus search 失败:{exc}") from exc
        hits = res[0] if res else []
        return [(_extract_pk(hit), float(hit["distance"])) for hit in hits]


def _is_released_error(exc: Exception) -> bool:
    """识别 Milvus「集合未加载」错(code=101 / released)。"""
    text = str(exc).lower()
    return "released" in text or "code=101" in text or "call load()" in text


def _extract_pk(hit) -> int:
    """从检索命中取主键。

    pymilvus 3.0 的命中按主键字段名暴露(fast-path 建集合时叫 chunk_id;
    默认 id);三种名字都兼容,取不到就抛错而不是静默错位。
    """
    for key in ("chunk_id", "id", "pk"):
        try:
            if key in hit:
                return int(hit[key])
        except (KeyError, TypeError, ValueError):
            continue
    raise VectorStoreError(f"检索命中缺少主键字段,keys={list(hit.keys())}")
