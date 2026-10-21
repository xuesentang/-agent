"""eval_runs 表：自动化评估流水线每轮的分数（ch09 DDL）。

一行 = 一轮评估。`metrics` 存整份指标 JSON，不拆列：不同策略、有没有跑生成段
需要记的东西不一样，拆成固定列就得为每种组合加列。

`triggered_by` 用中文枚举（定时 / 手动），与 DDL 一致。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt

TRIGGER_SCHEDULED = "定时"
TRIGGER_MANUAL = "手动"
TRIGGERS = (TRIGGER_SCHEDULED, TRIGGER_MANUAL)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(
        Enum(*TRIGGERS, name="triggered_by"), default=TRIGGER_SCHEDULED
    )
    dataset_size: Mapped[int] = mapped_column(Integer)
    # {"strategy": "hybrid_rerank", "recall_at_5": 0.84, "mrr": 0.664, "faithfulness": 0.9, ...}
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
