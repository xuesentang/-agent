"""messages 表:会话消息流水,role 对齐 Chat Completions 协议 (spec §5 DDL)。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInt, ForeignKey("conversations.id"), index=True
    )
    # user / assistant / tool — assistant 纯工具调用时 content 为 NULL、tool_calls 有值;
    # tool 消息 content 存结果、tool_call_id 对号入座 (回灌 LLM 时按 Chat Completions 协议重建)。
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="message_role"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
