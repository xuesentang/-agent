"""conversation_summaries 表:会话摘要分段(第七章需求 1、2)。

为什么单开一张表而不是塞进 messages:梗概既不是用户说的、也不是客服说的,
塞进逐条消息里,前端回载历史时用户会看到一段自己没说过的话。摘要段是「对一段
历史的压缩」,它有自己的边界(from_msg_id / to_msg_id)和顺序(seq),这些
在 messages 表里没有位置。

一段一条、只追加不重写:压完的梗概不参与下一轮合并,免得同一个事实被反复有损
压缩——第二次压缩读的是第一次的摘要,错一次就再也纠不回来。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigInt

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInt, ForeignKey("conversations.id"), index=True
    )
    # 第几段(从 1 起,同会话内递增),日志里「summary done 第N段」用的就是它
    seq: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[str] = mapped_column(Text)
    # 这段覆盖的消息 id 区间(闭区间,对应 messages.id)
    from_msg_id: Mapped[int] = mapped_column(BigInt, default=0)
    to_msg_id: Mapped[int] = mapped_column(BigInt, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="summaries")
