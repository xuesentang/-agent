"""low_confidence_questions 表:低置信度问题池 (ch04 DDL)。

三个入池入口:
- retrieval_low_conf 检索证据太弱(分数低于阈值/无命中),没进生成;
- self_check        生成前的证据充分性自评判定「不够答」;
- user_feedback     用户点了 👎(本章只做前端采集,后端入口留给后面)。

这张表是后面数据飞轮的原料:每一条都是「系统当时答不了」的真实问法。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInt, ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        String(24), default="retrieval_low_conf", index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
