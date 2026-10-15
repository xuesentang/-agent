"""tickets 表:人工工单,工单号当业务主键 (spec §5 DDL)。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Ticket(Base):
    __tablename__ = "tickets"

    # 工单号是业务主键(非自增 id),如 T20260701008,create_ticket 生成。
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInt, ForeignKey("conversations.id"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"))
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="tickets")
