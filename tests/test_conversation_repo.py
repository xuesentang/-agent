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
    # 延续 CH01 未知 session 语义:不报错,静默新建。
    conv = await conversation_repo.get_or_create(session, 99999, "demo-2")
    assert conv.id != 99999
    assert conv.user_id == "demo-2"
