"""low_confidence_questions 数据访问(ch04)。

入池一条低置信度问题:原话 + 来源入口 + 原因。查询侧只做计数/按来源列,
供复盘与后续飞轮使用。
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.low_confidence_question import LowConfidenceQuestion

SOURCES = ("retrieval_low_conf", "self_check", "user_feedback")


async def add_question(
    session: AsyncSession,
    *,
    raw_question: str,
    source: str,
    reason: str | None = None,
    conversation_id: int | None = None,
) -> LowConfidenceQuestion:
    """入池一条。source 非法直接抛,别让脏值进 ENUM。"""
    if source not in SOURCES:
        raise ValueError(f"未知入池入口:{source}")
    row = LowConfidenceQuestion(
        conversation_id=conversation_id,
        raw_question=raw_question,
        source=source,
        reason=reason,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def list_by_source(
    session: AsyncSession, source: str, limit: int = 100
) -> list[LowConfidenceQuestion]:
    stmt = (
        select(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.source == source)
        .order_by(LowConfidenceQuestion.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def count_all(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(LowConfidenceQuestion))).scalar() or 0
