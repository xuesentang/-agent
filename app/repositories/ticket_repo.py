"""tickets 表数据访问(spec §4 create_ticket 落地)。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ticket import Ticket


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

    注意:序号查询与插入在同一事务内,但两个并发请求可能同时读到同一 max
    (无行锁)——ch08 审计与权限章之前,以本注释显式标注该已知缺口,
    不在本章引入 SELECT ... FOR UPDATE(MySQL 方言)破坏 SQLite 单测。
    """
    date_str = datetime.now().strftime("%Y%m%d")
    seq = await next_ticket_seq(session, date_str)
    ticket = Ticket(
        ticket_no=f"T{date_str}{seq:04d}",
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    return ticket
