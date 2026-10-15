"""conversations 表数据访问(spec §5)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


async def get_or_create(
    session: AsyncSession, conversation_id: int | None, user_id: str
) -> Conversation:
    """按 id 取会话;id 为 None 或不存在时,复用该用户最近「进行中」的会话,
    没有才新建。

    语义:session_id 即用户键,连续发消息就是同一通对话(CH01 session 语义
    在 DB 上的落地);显式传 conversation_id 可续指任意历史会话;未知 id
    不报错,静默走复用/新建(CH01 未知 session 语义)。
    """
    if conversation_id is not None:
        existing = await session.get(Conversation, conversation_id)
        if existing is not None:
            return existing
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id, Conversation.status == "进行中")
        .order_by(Conversation.id.desc())
        .limit(1)
    )
    latest = (await session.execute(stmt)).scalar_one_or_none()
    if latest is not None:
        return latest
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv
