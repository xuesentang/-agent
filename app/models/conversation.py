"""conversations 表:会话壳,messages / tickets 都引用它 (spec §5 DDL)。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Enum, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt

if TYPE_CHECKING:
    from app.models.conversation_summary import ConversationSummary
    from app.models.message import Message
    from app.models.ticket import Ticket


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"), default="进行中"
    )
    # CH07 三段式的两个锚点(需求 1):降级只挪这两个 id,消息行永远不动。
    # summary_upto_msg_id = 摘要已覆盖到哪条 messages.id;layer1_from_msg_id = 层 1 从哪条起。
    summary_upto_msg_id: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    layer1_from_msg_id: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    summaries: Mapped[list["ConversationSummary"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
