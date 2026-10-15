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
from app.models.ticket import Ticket
from app.repositories import conversation_repo


def _call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


async def _db_tools(session_factory) -> dict:
    return {t.name: t for t in build_db_tools(session_factory)}


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


# ---------------------------------------------------------------- query_faq
async def test_query_faq_hit(session, session_factory):
    session.add(Faq(question="退货政策是什么", answer="签收后7天内可退", category="退换货"))
    await session.commit()
    tools = await _db_tools(session_factory)
    out = (await tools["query_faq"].ainvoke(_call("query_faq", {"keyword": "退货"}))).content
    assert "签收后7天内可退" in out
    assert "找到 1 条" in out


async def test_query_faq_miss_wording(session, session_factory):
    # 「邮费」不在种子数据里——ch02 预期漏召回样例(ch03 升级钩子)。
    tools = await _db_tools(session_factory)
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
