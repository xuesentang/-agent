"""按意图的 token 统计接口（ch09 需求 6）。

口径写在响应里（`token_source` / `note`）：这是估算值，能比「哪类意图烧钱」，
不能当账单。会上一旦有人拿它对账，问题就出在这里，所以宁可说三遍。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services import usage_service

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/by-intent")
async def usage_by_intent(session: AsyncSession = Depends(get_session)) -> dict:
    """各意图的 prompt / completion 总量、占比、轮均。空表照常 200。"""
    return await usage_service.by_intent(session)
