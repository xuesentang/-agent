"""low_confidence_questions 表:低置信度问题池 (ch04 DDL, ch09 加两列)。

三个入池入口:
- retrieval_low_conf 检索证据太弱(分数低于阈值/无命中),没进生成;
- self_check        生成前的证据充分性自评判定「不够答」;
- user_feedback     用户点了 👎(ch09 起接后端,ch04 时只是前端采集)。

这张表是数据飞轮的原料:每一条都是「系统当时答不了」的真实问法。

ch09 补的两列:
- `retrieved_chunks` 落池当时的召回片段快照(Top 几条原文与得分)。审核人靠它判断
  「知识库真缺这块」还是「有但没检到」——这两个结论对应的动作完全不同。
- `matched_review_id` 标准化查重后归并到的缺口行。原话行不删:一条缺口可能由十几条
  问法贡献,审核页要按这个外键反查「谁把它顶上来的」。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, Text, func
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
    # 落池时的召回片段快照:[{chunk_id, section_path, score, snippet}]，没走检索为 None
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # 归并到的缺口（review_queue.id）；None = 还没跑飞轮
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInt,
        ForeignKey("review_queue.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
