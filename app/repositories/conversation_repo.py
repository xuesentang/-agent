"""conversations 表数据访问(spec §5)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


async def get_or_create(
    session: AsyncSession, conversation_id: int | None, user_id: str
) -> Conversation:
    """按 id 取会话;id 为 None 或不存在时静默新建(延续 CH01 未知 session 语义)。

    user_id 仅在新建时使用(会话壳归属);已存在会话不改归属。
    """
    if conversation_id is not None:
        existing = await session.get(Conversation, conversation_id)
        if existing is not None:
            return existing
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv
