"""ORM models against SQLite in-memory (plan T1 Step 3)."""

from datetime import datetime

from sqlalchemy import text

from app.models import Conversation, Faq, Message, Ticket


async def test_create_all_four_tables(session):
    rows = (
        await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    ).scalars().all()
    assert {"conversations", "messages", "faq", "tickets"} <= set(rows)


async def test_conversation_defaults(session):
    conv = Conversation(user_id="demo-1")
    session.add(conv)
    await session.commit()
    assert conv.id is not None
    assert conv.status == "进行中"  # DDL DEFAULT
    assert isinstance(conv.created_at, datetime)


async def test_message_tool_calls_json_roundtrip(session):
    conv = Conversation(user_id="demo-1")
    session.add(conv)
    await session.commit()
    msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=None,  # assistant 纯工具调用时正文为空
        tool_calls=[
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    assert msg.tool_calls[0]["args"]["order_id"] == "1001"


async def test_message_tool_role_pairs_with_call_id(session):
    conv = Conversation(user_id="demo-1")
    session.add(conv)
    await session.commit()
    msg = Message(
        conversation_id=conv.id, role="tool", content="已到达北京转运中心", tool_call_id="call_1"
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"


async def test_ticket_business_primary_key(session):
    conv = Conversation(user_id="demo-1")
    session.add(conv)
    await session.commit()
    ticket = Ticket(
        ticket_no="T20260701008",
        conversation_id=conv.id,
        description="鞋子开胶要退货",
        ticket_type="售后",
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    assert ticket.ticket_no == "T20260701008"
    assert ticket.status == "待处理"  # DDL DEFAULT


async def test_faq_fields_roundtrip(session):
    faq = Faq(question="退货政策是什么", answer="签收后7天内可退", category="退换货")
    session.add(faq)
    await session.commit()
    await session.refresh(faq)
    assert faq.category == "退换货"
    assert isinstance(faq.created_at, datetime)
