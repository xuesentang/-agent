"""conversations 表数据访问(spec §5)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


async def list_by_user(session: AsyncSession, user_id: str, limit: int = 50) -> list[Conversation]:
    """该用户全部会话,新的在前(侧栏用)。"""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_or_create(
    session: AsyncSession,
    conversation_id: int | None,
    user_id: str,
    force_new: bool = False,
) -> Conversation:
    """按 id 取会话;id 为 None / 查无 / 不属于该用户时,复用该用户最近「进行中」的
    会话,没有才新建。force_new=True 跳过复用,直接新建(前端「新会话」按钮)。

    语义:session_id 即用户键,连续发消息就是同一通对话(CH01 session 语义
    在 DB 上的落地);显式传 conversation_id 可续指任意自己的历史会话;
    未知 id 或他人会话 id 不报错,静默走复用/新建(CH01 未知 session 语义)。
    """
    if conversation_id is not None and not force_new:
        existing = await session.get(Conversation, conversation_id)
        if existing is not None and existing.user_id == user_id:
            return existing
    if not force_new:
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
