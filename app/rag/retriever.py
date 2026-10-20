"""在线检索(ch03 T6):问题向量化 → Milvus Top-K → 回 MySQL 取原文。

query_faq 的内部实现从这里换核(CH02 的 LIKE 关键词查询降级为兜底),
工具入参出参契约不变。异常分流:
- 空结果 → 由 query_faq 回「未找到」文案(语义检索的诚实答案,不假装有);
- 向量库/嵌入异常 → VectorStoreError 等向上抛,由 query_faq 捕获降级 LIKE,
  检索器自身不做静默兜底(降级策略是工具层的事,这里只负责检索)。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.rag.vectorstore import MilvusStore


@dataclass(frozen=True)
class Hit:
    """一条检索命中:chunk 原文(MySQL 权威源)+ 相似度分。"""

    chunk: object  # KnowledgeChunk,不 import 避免循环依赖
    score: float


@dataclass(frozen=True)
class Retriever:
    """检索器 = 向量库 + 嵌入器,两者都可注入(单测替身 / 日后换实现)。"""

    store: MilvusStore
    embedder: object


async def retrieve(
    *,
    session_factory,
    query: str,
    top_k: int = 5,
    retriever: Retriever | None = None,
) -> list[Hit]:
    """向量检索主链路:embed(query) → Milvus search → 回 MySQL 取原文。

    返回按分数降序的 Hit 列表;空集合/无命中返回 []。
    Milvus 主键与 MySQL chunk id 严格对齐,回表后按检索分序重排
    (get_by_ids 按 id 升序返回,顺序以搜索为准)。
    """
    retriever = retriever or get_default_retriever()
    vector = retriever.embedder.embed_texts([query])[0]
    hits = retriever.store.search(vector, top_k)  # [(chunk_id, score)] 分降序
    if not hits:
        return []

    from app.repositories import knowledge_repo

    async with session_factory() as session:
        chunks = await knowledge_repo.get_by_ids(session, [cid for cid, _ in hits])
    by_id = {c.id: c for c in chunks}
    out: list[Hit] = []
    for chunk_id, score in hits:
        chunk = by_id.get(chunk_id)
        if chunk is not None:  # 权威源缺行时跳过,不让一条脏数据撑翻整条检索
            out.append(Hit(chunk=chunk, score=score))
    return out


@lru_cache(maxsize=1)
def get_default_retriever() -> Retriever:
    """进程内单例:真 Milvus(配置 URI)+ 真 BGE-M3。集合生命周期归
    build_knowledge 管,这里不 ensure——search 自带 released 态重试。"""
    from app.rag.embeddings import LocalBGEEmbedder

    return Retriever(store=MilvusStore(), embedder=LocalBGEEmbedder())


def warmup() -> None:
    """进程启动预热:加载嵌入模型(~2G 权重)+ 拉起向量库集合。

    不预热的话,首个 query_faq 要在工具超时(默认 10s)内背下模型加载,
    必超时——超时是执行层强制的,工具内部的降级兜底接不住它。
    预热失败不致命:记日志,首条查询现加载(可能超时降级 LIKE)。
    """
    retriever = get_default_retriever()
    retriever.store.ensure_collection()
    retriever.embedder.embed_texts(["预热"])
