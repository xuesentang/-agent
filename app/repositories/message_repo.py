"""messages 表数据访问(spec §5 / D6 全量落库)。"""

from __future__ import annotations

from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message

Role = Literal["user", "assistant", "tool"]


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    role: Role,
    content: str | None = None,
    tool_calls: dict | list | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """落一条消息并提交。assistant 纯工具调用时 content=None、tool_calls 有值;
    tool 消息 content=结果摘要、tool_call_id 对号入座。"""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


async def list_by_conversation(session: AsyncSession, conversation_id: int) -> list[Message]:
    """按 id 升序取全会话消息(时间序),回灌 LLM 时按此序重建。"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars().all())
