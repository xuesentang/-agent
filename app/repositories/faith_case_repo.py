"""faith_cases 数据访问(ch04 编造个案台账)。

一题一行:再次被判编造只刷快照 + seen_count+1;已解决的题复发要退回「未解决」。
报告是一次性的,台账才是跨轮资产——所以写入逻辑必须幂等且能识别复发。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.faith_case import FaithCase


async def upsert_case(
    session: AsyncSession,
    *,
    eval_id: str,
    bucket: str,
    query: str,
    answer: str,
    reason: str,
    strategy: str = "hybrid_rerank",
    citations: list | None = None,
    judge_model: str | None = None,
) -> tuple[FaithCase, bool]:
    """写入一条编造个案,返回 (行, 是否复发)。

    复发定义:这条之前是「已解决」/「无需解决」,又被判出来 → 退回「未解决」并清空
    resolution(resolved_at 保留,用来标复发)。
    """
    row = (
        await session.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))
    ).scalar_one_or_none()
    now = datetime.now()
    if row is None:
        row = FaithCase(
            eval_id=eval_id,
            bucket=bucket,
            query=query,
            strategy=strategy,
            answer=answer,
            reason=reason,
            citations=citations,
            judge_model=judge_model,
            status="未解决",
            seen_count=1,
            first_seen_at=now,
            last_seen_at=now,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row, False

    recurred = row.status in ("已解决", "无需解决")
    row.bucket = bucket
    row.query = query
    row.strategy = strategy
    row.answer = answer
    row.reason = reason
    row.citations = citations
    row.judge_model = judge_model
    row.seen_count += 1
    row.last_seen_at = now
    if recurred:
        row.status = "未解决"
        row.resolution = None
    await session.commit()
    await session.refresh(row)
    return row, recurred


async def list_open(session: AsyncSession, limit: int = 100) -> list[FaithCase]:
    """待处理(未解决)列表,按最近一次被判时间倒序。"""
    stmt = (
        select(FaithCase)
        .where(FaithCase.status == "未解决")
        .order_by(FaithCase.last_seen_at.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def resolve_case(
    session: AsyncSession, eval_id: str, status: str, resolution: str
) -> FaithCase | None:
    """人工处置:已解决/无需解决必须写处置说明(空着等于没交代)。"""
    if status not in ("已解决", "无需解决"):
        raise ValueError(f"非法处置状态:{status}")
    if not resolution.strip():
        raise ValueError("处置说明不能为空")
    row = (
        await session.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))
    ).scalar_one_or_none()
    if row is None:
        return None
    row.status = status
    row.resolution = resolution
    row.resolved_at = datetime.now()
    await session.commit()
    await session.refresh(row)
    return row
