"""ticket_repo TDD(spec Review Focus 3:工单号格式与当日序列)。"""

import re

from app.repositories import conversation_repo, ticket_repo


async def _new_conversation(session):
    return await conversation_repo.get_or_create(session, None, "demo-1")


async def test_ticket_no_format(session):
    conv = await _new_conversation(session)
    ticket = await ticket_repo.create(session, conv.id, "鞋子开胶要退货", "售后")
    assert re.fullmatch(r"T\d{8}\d{4}", ticket.ticket_no)
    assert ticket.status == "待处理"
    assert ticket.conversation_id == conv.id


async def test_seq_increments_same_day(session):
    conv = await _new_conversation(session)
    t1 = await ticket_repo.create(session, conv.id, "问题一", "售后")
    t2 = await ticket_repo.create(session, conv.id, "问题二", "投诉")
    assert t1.ticket_no[:9] == t2.ticket_no[:9]  # 同日
    assert int(t2.ticket_no[-4:]) == int(t1.ticket_no[-4:]) + 1
    assert t2.ticket_type == "投诉"
