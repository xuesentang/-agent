"""tickets 表数据访问(spec §4 create_ticket 落地)。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ticket import Ticket

MAX_SEQ_RETRIES = 3


async def next_ticket_seq(session: AsyncSession, date_str: str) -> int:
    """当日工单序列:已有最大序号 + 1,没有则 1。"""
    prefix = f"T{date_str}"
    stmt = (
        select(Ticket.ticket_no)
        .where(Ticket.ticket_no.like(f"{prefix}%", escape="\\"))
        .order_by(Ticket.ticket_no.desc())
        .limit(1)
    )
    last = (await session.execute(stmt)).scalar_one_or_none()
    return (int(last[-4:]) + 1) if last else 1


async def create(
    session: AsyncSession,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Ticket:
    """生成工单号 T{yyyyMMdd}{4位序列} 并落库。

    并发撞号窗口(review Minor 6):序号查询无行锁,两个并发请求可能同时读到
    同一 max。ch08 权限与审计章之前不引入 SELECT ... FOR UPDATE(MySQL 方言,
    会破坏 SQLite 单测),改为 IntegrityError → seq+1 重试,成本极低。
    """
    date_str = datetime.now().strftime("%Y%m%d")
    for attempt in range(MAX_SEQ_RETRIES):
        seq = await next_ticket_seq(session, date_str) + attempt
        ticket = Ticket(
            ticket_no=f"T{date_str}{seq:04d}",
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        session.add(ticket)
        try:
            await session.commit()
            await session.refresh(ticket)
            return ticket
        except IntegrityError:
            await session.rollback()
            continue  # 撞号:下一个序列再试
    raise RuntimeError(f"工单号生成重试 {MAX_SEQ_RETRIES} 次仍冲突,请稍后重试")
