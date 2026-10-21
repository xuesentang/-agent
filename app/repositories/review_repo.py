"""review_queue 数据访问（ch09）。

三条不变量，写在这里防止后面改坏：

1. **查重命中只累加，不新建行**：`bump()` 是唯一增加 `occurrence_count` 的入口，
   它不碰 `normalized_question`——问题文本被反复改写会让同一缺口前后不一致。
2. **状态流转单向**：待审 → 通过 / 驳回。已通过的缺口不能再改回待审，
   知识库里已经有答案了，改状态只会让「通过但不在库里」这种脏数据出现。
3. **答案与状态一起改**：`mark_approved()` 一次写 `approved_answer` + 状态，
   分两次写会出现「状态是通过、答案是空」的中间态。
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.review_queue import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    ReviewQueue,
)


async def create(
    session: AsyncSession,
    *,
    normalized_question: str,
    ai_suggested_answer: str | None = None,
    commit: bool = True,
) -> ReviewQueue:
    row = ReviewQueue(
        normalized_question=normalized_question.strip()[:512],
        ai_suggested_answer=ai_suggested_answer,
        occurrence_count=1,
        review_status=STATUS_PENDING,
    )
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def bump(session: AsyncSession, review_id: int, *, commit: bool = True) -> ReviewQueue | None:
    """命中已有缺口：出现次数 +1，返回该行（不存在返回 None）。"""
    row = await session.get(ReviewQueue, review_id)
    if row is None:
        return None
    row.occurrence_count = int(row.occurrence_count or 0) + 1
    if commit:
        await session.commit()
        await session.refresh(row)
    return row


async def get(session: AsyncSession, review_id: int) -> ReviewQueue | None:
    return await session.get(ReviewQueue, review_id)


async def list_by_status(
    session: AsyncSession, status: str = STATUS_PENDING, limit: int = 100
) -> list[ReviewQueue]:
    """按出现次数排（越靠前越该优先补），同次数按入队先后。"""
    stmt = (
        select(ReviewQueue)
        .where(ReviewQueue.review_status == status)
        .order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.id.asc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_for_dedupe(session: AsyncSession, limit: int = 20) -> list[ReviewQueue]:
    """查重候选池：只和**待审**的比。

    已通过的有答案在库里了；再让新问题并进去只会加次数，不会产生新知识。
    已驳回的也不该再吸新问题——驳回通常意味着「这不是知识缺口」。
    """
    return await list_by_status(session, STATUS_PENDING, limit=limit)


async def mark_approved(
    session: AsyncSession, review_id: int, approved_answer: str, *, commit: bool = True
) -> ReviewQueue | None:
    row = await session.get(ReviewQueue, review_id)
    if row is None:
        return None
    row.review_status = STATUS_APPROVED
    row.approved_answer = approved_answer.strip()
    if commit:
        await session.commit()
        await session.refresh(row)
    return row


async def mark_rejected(
    session: AsyncSession, review_id: int, *, commit: bool = True
) -> ReviewQueue | None:
    row = await session.get(ReviewQueue, review_id)
    if row is None:
        return None
    row.review_status = STATUS_REJECTED
    if commit:
        await session.commit()
        await session.refresh(row)
    return row


async def counts(session: AsyncSession) -> dict[str, int]:
    """三个状态各多少条（后台页头部的数字）。"""
    rows = (
        await session.execute(
            select(ReviewQueue.review_status, func.count()).group_by(ReviewQueue.review_status)
        )
    ).all()
    return {status: int(count) for status, count in rows}
