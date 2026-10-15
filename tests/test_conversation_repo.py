"""conversation_repo TDD(spec Review Focus 1:未知 id 静默新建)。"""

from app.repositories import conversation_repo


async def test_create_when_id_none(session):
    conv = await conversation_repo.get_or_create(session, None, "demo-1")
    assert conv.id is not None
    assert conv.user_id == "demo-1"
    assert conv.status == "进行中"


async def test_get_existing(session):
    conv = await conversation_repo.get_or_create(session, None, "demo-1")
    same = await conversation_repo.get_or_create(session, conv.id, "demo-1")
    assert same.id == conv.id


async def test_unknown_id_creates_silently(session):
    # 延续 CH01 未知 session 语义:不报错,静默走复用/新建。
    conv = await conversation_repo.get_or_create(session, 99999, "demo-2")
    assert conv.id != 99999
    assert conv.user_id == "demo-2"


async def test_none_id_reuses_latest_active_conversation(session):
    """session_id 连续发消息即同一通对话(CH01 语义):不传 id 复用最近进行中会话。"""
    conv1 = await conversation_repo.get_or_create(session, None, "demo-1")
    conv2 = await conversation_repo.get_or_create(session, None, "demo-1")
    assert conv2.id == conv1.id
    # 别的用户不受影响
    other = await conversation_repo.get_or_create(session, None, "demo-2")
    assert other.id != conv1.id


async def test_ended_conversation_not_reused(session):
    conv1 = await conversation_repo.get_or_create(session, None, "demo-1")
    conv1.status = "已结束"
    await session.commit()
    conv2 = await conversation_repo.get_or_create(session, None, "demo-1")
    assert conv2.id != conv1.id
