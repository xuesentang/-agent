"""摘要表与锚点推进单测。"""

from __future__ import annotations

from app.repositories import conversation_repo, conversation_summary_repo, message_repo


async def test_summary_segments_append_in_order(session):
    conv = await conversation_repo.get_or_create(session, None, "u1")
    first = await conversation_summary_repo.add_summary(
        session, conv.id, seq=1, content="用户问过 W40 的尺码,报了订单 1001。", from_msg_id=1, to_msg_id=8
    )
    second = await conversation_summary_repo.add_summary(
        session, conv.id, seq=2, content="用户要求退货,原因质量问题。", from_msg_id=9, to_msg_id=16, tokens=30
    )
    rows = await conversation_summary_repo.list_by_conversation(session, conv.id)
    assert [r.seq for r in rows] == [1, 2]
    assert rows[0].id == first.id and rows[1].id == second.id
    assert (await conversation_summary_repo.next_seq(session, conv.id)) == 3


async def test_joined_text_keeps_segment_order(session):
    conv = await conversation_repo.get_or_create(session, None, "u2")
    await conversation_summary_repo.add_summary(session, conv.id, seq=1, content="第一段", from_msg_id=1, to_msg_id=4)
    await conversation_summary_repo.add_summary(session, conv.id, seq=2, content="第二段", from_msg_id=5, to_msg_id=9)
    assert await conversation_summary_repo.joined_text(session, conv.id) == "第一段\n第二段"


async def test_advance_anchors_moves_both_ids_together(session):
    conv = await conversation_repo.get_or_create(session, None, "u3")
    assert conv.summary_upto_msg_id == 0 and conv.layer1_from_msg_id == 0
    await conversation_summary_repo.advance_anchors(
        session, conv, summary_upto_msg_id=8, layer1_from_msg_id=9
    )
    assert conv.summary_upto_msg_id == 8
    assert conv.layer1_from_msg_id == 9


async def test_messages_are_never_moved_by_summary(session):
    """摘要只动锚点,消息行原样保留——前端回载旧会话要看到原文。"""
    conv = await conversation_repo.get_or_create(session, None, "u4")
    for i in range(6):
        await message_repo.add_message(session, conv.id, role="user" if i % 2 == 0 else "assistant", content=f"第{i}条")
    before = [(m.id, m.content) for m in await message_repo.list_by_conversation(session, conv.id)]
    await conversation_summary_repo.add_summary(
        session, conv.id, seq=1, content="梗概", from_msg_id=before[0][0], to_msg_id=before[3][0]
    )
    await conversation_summary_repo.advance_anchors(
        session, conv, summary_upto_msg_id=before[3][0], layer1_from_msg_id=before[4][0]
    )
    after = [(m.id, m.content) for m in await message_repo.list_by_conversation(session, conv.id)]
    assert before == after
