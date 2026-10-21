"""review_queue 表：飞轮待审队列（ch09 DDL）。

一行 = 一个**去重后的知识缺口**。同义问题查重命中时只累加 `occurrence_count`，
不新建行——所以「出现次数」是按缺口算的，不是按原话条数算的。

`review_status` 三态与 ch04 的入池 `source` 一样是业务枚举，直接用中文值，
和 `各章建表.md` 的 ENUM 一致；中文值出现在 SQL 里比数字好排查。

为什么原话不放在这张表：一条缺口可能被 20 条不同问法命中，原话留在
`low_confidence_questions` 里靠 `matched_review_id` 指过来。审核页要看的
「这条缺口由哪些真实问法贡献」就是按那个外键反查。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt

# 审核状态：字符串常量集中在这里，服务层/接口/前端都从这里取，不各写一份
STATUS_PENDING = "待审"
STATUS_APPROVED = "通过"
STATUS_REJECTED = "驳回"
STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    review_status: Mapped[str] = mapped_column(
        Enum(*STATUSES, name="review_status"), default=STATUS_PENDING, index=True
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
