"""faith_cases 表:忠实度编造个案台账 (ch04 DDL)。

一题一行(uk_eval_id):同一道题再次被判编造不新增行,只刷快照并 seen_count+1;
已标「已解决」的题又被判出来 → 状态退回「未解决」(复发不是新问题,得重新进待处理)。
citations 存这一轮喂给模型的 Top-K 证据全集,便于追溯「手里有哪几条、实际引了哪几条」。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_status"),
        default="未解决",
        index=True,
    )
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
