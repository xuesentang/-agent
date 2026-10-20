"""qa_extraction_staging 表:历史对话抽 QA 的离线中转暂存表 (ch03 DDL)。

流程(miner):messages 历史分批喂 LLM 抽 QA 对 → 本表 status=extracted →
整体去重 → kept 项入 knowledge_chunks / discarded 标注;建库完成可清空。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_staging_status"),
        default="extracted",
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
