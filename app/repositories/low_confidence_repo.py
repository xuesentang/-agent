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
    retrieved_chunks: list | None = None,
) -> LowConfidenceQuestion:
    """入池一条。source 非法直接抛,别让脏值进 ENUM。"""
    if source not in SOURCES:
        raise ValueError(f"未知入池入口:{source}")
    row = LowConfidenceQuestion(
        conversation_id=conversation_id,
        raw_question=raw_question,
        source=source,
        reason=reason,
        retrieved_chunks=retrieved_chunks,
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


async def list_unmerged(
    session: AsyncSession, limit: int = 50, since_id: int = 0
) -> list[LowConfidenceQuestion]:
    """还没被飞轮处理过的池内问题（`matched_review_id IS NULL`）。

    按 id 升序：老问题先处理，免得新问题一直插队、老问题永远排在后面。
    `since_id` 给「续跑」用（上一批跑完停在哪一行）。
    """
    stmt = (
        select(LowConfidenceQuestion)
        .where(
            LowConfidenceQuestion.matched_review_id.is_(None),
            LowConfidenceQuestion.id > since_id,
        )
        .order_by(LowConfidenceQuestion.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def set_matched_review(
    session: AsyncSession, question_id: int, review_id: int, *, commit: bool = True
) -> bool:
    """记下这条原话归并到哪个缺口。返回是否命中到行。"""
    row = await session.get(LowConfidenceQuestion, question_id)
    if row is None:
        return False
    row.matched_review_id = review_id
    if commit:
        await session.commit()
    return True


async def list_by_review(
    session: AsyncSession, review_id: int, limit: int = 100
) -> list[LowConfidenceQuestion]:
    """某缺口由哪些真实问法贡献（审核页详情要看的原文）。

    新的在前：审核人先看到最近被顶上来的是怎么说的。
    """
    stmt = (
        select(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.matched_review_id == review_id)
        .order_by(LowConfidenceQuestion.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())
