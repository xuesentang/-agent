"""五个业务工具 TDD(spec §4 / Review Focus 6:mock 确定性)。"""

from sqlalchemy import func, select

from app.core.tools.business_tools import (
    build_db_tools,
    build_default_registry,
    query_logistics,
    query_order,
    query_product,
)
from app.core.tools.context import current_conversation_id
from app.models.faq import Faq
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.ticket import Ticket
from app.rag.embeddings import FakeEmbedder
from app.rag.retriever import Retriever
from app.rag.vectorstore import FakeMilvus, MilvusStore, VectorStoreError
from app.repositories import conversation_repo, knowledge_repo


def _call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


async def _db_tools(session_factory, retriever: Retriever | None = None) -> dict:
    return {t.name: t for t in build_db_tools(session_factory, retriever)}


def _fake_retriever(chunks: list[KnowledgeChunk] | None = None) -> Retriever:
    """FakeMilvus + FakeEmbedder 检索器;chunks 灌进去当知识库。"""
    client = FakeMilvus()
    store = MilvusStore(client=client)
    embedder = FakeEmbedder()
    if chunks:
        store.upsert(
            [
                {"chunk_id": c.id, "embedding": embedder.embed_texts([c.embedding_text()])[0]}
                for c in chunks
            ]
        )
    return Retriever(store=store, embedder=embedder)


async def _seed_chunk(session, answer: str, questions: str = "测试问题") -> KnowledgeChunk:
    chunk = KnowledgeChunk(category="测试", questions=questions, answer=answer)
    await knowledge_repo.add_chunks(session, [chunk])
    return chunk


# ---------------------------------------------------------------- mock 三工具
async def test_query_order_deterministic_and_shaped():
    out = (await query_order.ainvoke(_call("query_order", {"order_id": "1001"}))).content
    assert "1001" in out
    assert "¥" in out
    assert "状态" in out
    again = (await query_order.ainvoke(_call("query_order", {"order_id": "1001"}))).content
    assert out == again  # 同订单同结果


async def test_query_order_varies_by_order_id():
    a = (await query_order.ainvoke(_call("query_order", {"order_id": "1001"}))).content
    b = (await query_order.ainvoke(_call("query_order", {"order_id": "1002"}))).content
    assert a != b


async def test_query_logistics_deterministic_and_trajectory_shape():
    out = (await query_logistics.ainvoke(_call("query_logistics", {"order_id": "1001"}))).content
    assert out.count("\n- ") >= 3  # 至少 3 个节点
    assert "1001" in out
    assert "最新在上" in out
    assert out == (await query_logistics.ainvoke(_call("query_logistics", {"order_id": "1001"}))).content


async def test_query_product_deterministic_and_shaped():
    out = (await query_product.ainvoke(_call("query_product", {"product_name": "跑鞋"}))).content
    assert "跑鞋" in out
    assert "¥" in out
    assert "库存" in out
    assert out == (await query_product.ainvoke(_call("query_product", {"product_name": "跑鞋"}))).content


# ---------------------------------------------------------------- query_faq(ch03:向量优先,LIKE 兜底)
async def test_query_faq_vector_hit(session, session_factory):
    """向量路命中:输出格式与 CH02 逐字一致(找到 N 条 / 问:答:)。"""
    chunk = await _seed_chunk(session, "签收后 7 天内可退,吊牌完好。", questions="退货政策是什么")
    tools = await _db_tools(session_factory, _fake_retriever([chunk]))
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "退货"}))).content
    assert "找到 1 条相关 FAQ:" in out
    assert "问:退货政策是什么" in out
    assert "答:签收后 7 天内可退,吊牌完好。" in out


async def test_query_faq_vector_miss_wording(session_factory):
    """向量路空结果:未找到文案(语义检索的诚实答案,不降级 LIKE)。"""
    tools = await _db_tools(session_factory, _fake_retriever())
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "邮费"}))).content
    assert "未在 FAQ 知识库中找到" in out


async def test_query_faq_store_error_falls_back_to_like(session, session_factory):
    """Milvus 故障 → 降级 CH02 的 LIKE,输出同款(徽章/回灌格式不变)。"""
    session.add(Faq(question="退货政策是什么", answer="签收后7天内可退", category="退换货"))
    await session.commit()

    class BoomClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise VectorStoreError("milvus down")

    retriever = Retriever(store=MilvusStore(client=BoomClient()), embedder=FakeEmbedder())
    tools = await _db_tools(session_factory, retriever)
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "退货"}))).content
    assert "找到 1 条相关 FAQ:" in out
    assert "答:签收后7天内可退" in out


async def test_query_faq_embedder_error_falls_back_to_like(session, session_factory):
    """嵌入故障同样降级 LIKE。"""
    session.add(Faq(question="运费怎么算", answer="满 99 包邮", category="运费"))
    await session.commit()

    class BoomEmbedder:
        def embed_texts(self, texts):
            raise RuntimeError("model crashed")

    retriever = Retriever(store=MilvusStore(client=FakeMilvus()), embedder=BoomEmbedder())
    tools = await _db_tools(session_factory, retriever)
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "运费"}))).content
    assert "答:满 99 包邮" in out


async def test_query_faq_fallback_miss_wording(session_factory):
    """降级路也没命中:同样的未找到文案。"""

    class BoomClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise VectorStoreError("milvus down")

    retriever = Retriever(store=MilvusStore(client=BoomClient()), embedder=FakeEmbedder())
    tools = await _db_tools(session_factory, retriever)
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "邮费"}))).content
    assert "未在 FAQ 知识库中找到" in out


# ---------------------------------------------------------------- create_ticket
async def test_create_ticket_requires_conversation_context(session_factory):
    tools = await _db_tools(session_factory)
    out = (
        await tools["create_ticket"].ainvoke(
            _call("create_ticket", {"description": "鞋子开胶", "ticket_type": "售后"})
        )
    ).content
    assert "缺少会话上下文" in out


async def test_create_ticket_with_context_writes_row(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "demo-1")
    token = current_conversation_id.set(conv.id)
    try:
        tools = await _db_tools(session_factory)
        out = (
            await tools["create_ticket"].ainvoke(
                _call(
                    "create_ticket",
                    {"description": "鞋子穿两天开胶,要退货退款", "ticket_type": "售后"},
                )
            )
        ).content
        assert "已创建工单 T" in out
        assert "售后" in out
        count = (await session.execute(select(func.count()).select_from(Ticket))).scalar()
        assert count == 1
    finally:
        current_conversation_id.reset(token)


# ---------------------------------------------------------------- 全局清单
def test_default_registry_has_five_tools():
    reg = build_default_registry()
    assert set(reg.names()) == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
