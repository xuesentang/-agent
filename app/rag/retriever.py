"""在线检索(ch03 换核 / ch04 混合检索 + 重排)。

四策略统一入口(ch04 spec D1):
- dense        纯向量(COSINE);
- bm25         纯关键词(Milvus 原生 BM25 + jieba 分词);
- hybrid       两路各 Top-50 → hybrid_search 配 RRF 融合;
- hybrid_rerank 混合粗排 Top-20 → bge-reranker-v2-m3 精排 → Top-K。

**默认走 dense**(2026-10-20 四策略对照的结论,见 dev-notes/ch04.md「定策略」):
357 块库上 dense 的 R@5 0.81 / R@10 0.94 / MRR 0.709,hybrid 虽然 R@5 高 0.03
但 MRR 低 0.155,hybrid_rerank 把 MRR 救回 0.664 却把 R@10 压到 0.90,还要多付
2.27G 权重与 CPU 精排。三条都保留可切换,别再默认挂重排。

理解层在检索侧:口语问法先归一(canonical)喂向量与关键词,同义词只扩进 BM25 查询,
入库侧一个字不改(见 query_rewrite 的模块注释)。

异常分流:空结果返回 [] (由上层决定「未找到」话术);向量库/嵌入异常向上抛,
由 query_faq 捕获降级 LIKE——检索器自身不静默兜底。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.rag.vectorstore import MilvusStore

logger = logging.getLogger(__name__)

# 类目名白名单:中文、字母、数字,以及空格/中点/斜杠/连字符(如「售后·退换 / 维修」)。
_SAFE_CATEGORY = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff ·/、-]*$")

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
DEFAULT_STRATEGY = "dense"
DEFAULT_TOP_K = 10
RERANK_POOL = 20  # 进重排的粗排候选数(ch04:精排出 Top-10)


@dataclass(frozen=True)
class Hit:
    """一条检索命中:chunk 原文(MySQL 权威源)+ 分数(策略不同量纲不同)。"""

    chunk: object  # KnowledgeChunk,不 import 避免循环依赖
    score: float


@dataclass(frozen=True)
class Retriever:
    """检索器 = 向量库 + 嵌入器 + 重排器,三者都可注入(单测替身 / 日后换实现)。"""

    store: MilvusStore
    embedder: object
    reranker: object | None = None


def _category_expr(category: str | None) -> str | None:
    """元数据过滤表达式:先过滤再检索(ch04 需求 3)。

    category 目前只来自内部映射,但它将来可能接工具入参(模型可填),
    所以按白名单校验:中文/字母/数字/常见分隔符以外的字符直接拒绝,
    不做「转义后放行」——Milvus 表达式没有可靠的引号转义约定。
    """
    if not category:
        return None
    if not _SAFE_CATEGORY.match(category):
        raise ValueError(f"非法类目名:{category!r}")
    return f'category == "{category}"'


async def retrieve(
    *,
    session_factory,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = DEFAULT_STRATEGY,
    category: str | None = None,
    retriever: Retriever | None = None,
    rewrite: bool = True,
    rewrite_model=None,
) -> list[Hit]:
    """检索主链路。返回按相关度降序的 Hit 列表;空集合/无命中返回 []。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略:{strategy}")
    retriever = retriever or get_default_retriever()

    rewritten = await _rewrite(query, rewrite=rewrite, model=rewrite_model)
    expr = _category_expr(category)

    if strategy == "bm25":
        raw = retriever.store.search_sparse(rewritten.bm25_query, top_k, expr=expr)
        return await _hydrate(session_factory, raw)

    vector = retriever.embedder.embed_texts([rewritten.canonical])[0]
    if strategy == "dense":
        raw = retriever.store.search(vector, top_k, expr=expr)
        return await _hydrate(session_factory, raw)

    if strategy == "hybrid":
        raw = retriever.store.hybrid_search(
            vector, rewritten.bm25_query, top_k, expr=expr, route_limit=_route_limit()
        )
        return await _hydrate(session_factory, raw)

    # hybrid_rerank:粗排多取一些,精排后再截断
    pool = max(top_k, RERANK_POOL)
    raw = retriever.store.hybrid_search(
        vector, rewritten.bm25_query, pool, expr=expr, route_limit=_route_limit()
    )
    hits = await _hydrate(session_factory, raw)
    if not hits or retriever.reranker is None:
        return hits[:top_k]
    scores = retriever.reranker.rerank(rewritten.canonical, [_chunk_text(h) for h in hits])
    reranked = [Hit(chunk=h.chunk, score=float(s)) for h, s in zip(hits, scores)]
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked[:top_k]


def _route_limit() -> int:
    """两路各召回多少条进 RRF。按库规模配(recall_route_limit),不写死。"""
    from app.config import get_settings

    return max(1, get_settings().recall_route_limit)


async def _rewrite(query: str, *, rewrite: bool, model):
    """理解层:没开改写就只做同义词扩展(不调模型)。"""
    from app.rag.query_rewrite import rewrite_query

    if not rewrite:
        return await rewrite_query(query, model=None)
    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001 - 拿不到模型就退回纯同义词
            logger.warning("改写模型不可用,退回同义词扩展:%s", exc)
            return await rewrite_query(query, model=None)
    return await rewrite_query(query, model=model)


def _chunk_text(hit: Hit) -> str:
    chunk = hit.chunk
    embed = getattr(chunk, "embedding_text", None)
    return embed() if callable(embed) else str(chunk)


async def _hydrate(session_factory, raw: list[tuple[int, float]]) -> list[Hit]:
    """Milvus 主键 → MySQL 原文,按检索分序重排(get_by_ids 按 id 升序返回)。"""
    if not raw:
        return []
    from app.repositories import knowledge_repo

    async with session_factory() as session:
        chunks = await knowledge_repo.get_by_ids(session, [cid for cid, _ in raw])
    by_id = {c.id: c for c in chunks}
    out: list[Hit] = []
    for chunk_id, score in raw:
        chunk = by_id.get(chunk_id)
        if chunk is not None:  # 权威源缺行时跳过,不让一条脏数据撑翻整条检索
            out.append(Hit(chunk=chunk, score=score))
    return out


@lru_cache(maxsize=1)
def get_default_retriever() -> Retriever:
    """进程内单例:真 Milvus(配置 URI)+ 真 BGE-M3 + 真 bge-reranker-v2-m3。"""
    from app.rag.embeddings import LocalBGEEmbedder
    from app.rag.reranker import get_default_reranker

    return Retriever(
        store=MilvusStore(),
        embedder=LocalBGEEmbedder(),
        reranker=get_default_reranker(),
    )


def warmup() -> None:
    """进程启动预热:加载嵌入模型(~2G)+ 重排模型(~2G)+ 拉起向量库集合。

    不预热的话,首个 query_faq 要在工具超时(默认 10s)内背下模型加载,必超时——
    超时是执行层强制的,工具内部的降级兜底接不住它。预热失败不致命:记日志,
    首条查询现加载(可能超时降级 LIKE)。
    """
    retriever = get_default_retriever()
    retriever.store.ensure_collection()
    retriever.embedder.embed_texts(["预热"])
    if retriever.reranker is not None:
        retriever.reranker.rerank("预热", ["预热文本"])


def _traced(fn):
    """启用 Langfuse 时给检索包一层 retriever span,否则原样返回。

    SDK 的 LangChain 集成只覆盖 LangChain 世界(图节点 / LLM / 工具),
    检索是项目自己的代码,不埋点 trace 里就没有这一环——而检索恰恰是
    客服答不上问题时最该看的那一环。没装 SDK / 没配 key 时零开销。
    """
    settings = get_settings()
    if not (
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    ):
        return fn
    try:
        from langfuse import observe

        return observe(fn, name="rag.retrieve", as_type="retriever")
    except Exception:
        logger.warning("Langfuse observe 包装失败,检索不上报 span", exc_info=True)
        return fn


retrieve = _traced(retrieve)
