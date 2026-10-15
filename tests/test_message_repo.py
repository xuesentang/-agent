"""message_repo TDD(spec Review Focus 2:tool_calls 对号入座)。"""

from app.repositories import message_repo


async def _new_conversation(session):
    from app.repositories import conversation_repo

    return await conversation_repo.get_or_create(session, None, "demo-1")


async def test_add_and_list_in_order(session):
    conv = await _new_conversation(session)
    await message_repo.add_message(session, conv.id, "user", "订单1001的物流到哪了")
    await message_repo.add_message(
        session,
        conv.id,
        "assistant",
        content=None,
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1", "type": "tool_call"}],
    )
    await message_repo.add_message(
        session, conv.id, "tool", content="已到达北京转运中心", tool_call_id="call_1"
    )
    msgs = await message_repo.list_by_conversation(session, conv.id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1].tool_calls[0]["id"] == "call_1"
    assert msgs[2].tool_call_id == "call_1"  # 对号入座


async def test_list_only_own_conversation(session):
    from app.repositories import conversation_repo

    conv1 = await _new_conversation(session)
    conv2 = await conversation_repo.get_or_create(session, None, "demo-2")
    await message_repo.add_message(session, conv1.id, "user", "a")
    await message_repo.add_message(session, conv2.id, "user", "b")
    assert len(await message_repo.list_by_conversation(session, conv1.id)) == 1
