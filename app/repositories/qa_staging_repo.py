"""qa_extraction_staging 表数据访问(ch03 挖知识中转暂存表)。

miner 的写入与去重标注:抽出的 QA 对先落 extracted,去重后改 kept / discarded。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.qa_extraction_staging import QaExtractionStaging


async def add_batch(
    session: AsyncSession,
    batch_no: str,
    pairs: list[tuple[str, str]],
    source_ref: str | None = None,
    *,
    commit: bool = True,
) -> list[QaExtractionStaging]:
    """一批抽出的 QA 对落 staging(status=extracted)。pairs: [(question, answer)]。

    commit=False 时只 flush 不提交(挖知识要把暂存+翻状态+知识入库收进一个事务)。
    """
    rows = [
        QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=q, answer=a
        )
        for q, a in pairs
    ]
    session.add_all(rows)
    if commit:
        await session.commit()
    else:
        await session.flush()
    for row in rows:
        await session.refresh(row)
    return rows


async def list_by_status(
    session: AsyncSession, status: str, batch_no: str | None = None
) -> list[QaExtractionStaging]:
    """按状态列暂存行(去重遍历 extracted 用);可按批次过滤。"""
    stmt = select(QaExtractionStaging).where(QaExtractionStaging.status == status)
    if batch_no is not None:
        stmt = stmt.where(QaExtractionStaging.batch_no == batch_no)
    stmt = stmt.order_by(QaExtractionStaging.id)
    return list((await session.execute(stmt)).scalars().all())


async def set_status(session: AsyncSession, row_id: int, status: str) -> None:
    """改单行状态(kept / discarded)。"""
    row = await session.get(QaExtractionStaging, row_id)
    if row is None:
        return
    row.status = status
    await session.commit()


async def set_status_bulk(
    session: AsyncSession, ids: list[int], status: str, *, commit: bool = True
) -> None:
    """批量改状态(去重后一次提交;commit=False 交由调用方统一提交)。"""
    if not ids:
        return
    stmt = select(QaExtractionStaging).where(QaExtractionStaging.id.in_(ids))
    rows = (await session.execute(stmt)).scalars().all()
    for row in rows:
        row.status = status
    if commit:
        await session.commit()
    else:
        await session.flush()
