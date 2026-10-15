"""faq 表数据访问(spec §4 query_faq 的数据源)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.faq import Faq


def escape_like(term: str) -> str:
    """转义 LIKE 特殊字符:用户输入含 % _ \\ 时按字面匹配,防误匹配与注入。"""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def search_by_keyword(session: AsyncSession, keyword: str, limit: int = 5) -> list[Faq]:
    """SQL LIKE 关键词查 faq 表(ch02 关键词路线;ch03 起换向量检索)。

    命中 question 或 answer 任一含关键词的行,按 id 升序(稳定、可复现)。
    无命中返回空列表——调用方(工具)负责组织「未找到」话术。
    """
    pattern = f"%{escape_like(keyword)}%"
    stmt = (
        select(Faq)
        .where(
            Faq.question.like(pattern, escape="\\") | Faq.answer.like(pattern, escape="\\")
        )
        .order_by(Faq.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())
