"""审核队列接口（ch09 需求 4/6，后台页 `/admin/review` 的数据侧）。

三个约束：

- **读写分离**：GET 列表/详情，POST 通过/驳回。没有「改一改队列里的问题文本」这种入口——
  标准化问题被人工改了之后，和池里原话的对应关系就断了。
- **列表只读库**：GET 不加载嵌入模型；只有 approve 会（写知识库要算向量）。
  审核人翻十条队列不该触发一次 2G 的模型加载。
- **空队列照常 200**：还没跑飞轮时队列是空的，前端显示「先跑 scripts/run_flywheel.py」，
  不该给 404 让人以为接口挂了。
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.repositories import low_confidence_repo, review_repo
from app.services import eval_service, flywheel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/review", tags=["review"])

EMPTY_HINT = "队列还没建起来，先跑 python scripts/run_flywheel.py --batch 50"


class ApproveRequest(BaseModel):
    answer: str = Field(min_length=1, description="核准答案，会写进知识库")


@lru_cache(maxsize=1)
def _index_assets():
    """审核通过要用的嵌入器与向量库，进程内只建一次。

    优先复用检索器那份（启动预热已经建好了）：再 new 一个 LocalBGEEmbedder 就是
    在内存里放第二份 BGE-M3，开发台那点内存经不起。
    """
    from app.rag.retriever import get_default_retriever

    retriever = get_default_retriever()
    return retriever.embedder, retriever.store

def _review_payload(row) -> dict:
    return {
        "id": row.id,
        "normalized_question": row.normalized_question,
        "ai_suggested_answer": row.ai_suggested_answer,
        "occurrence_count": row.occurrence_count,
        "review_status": row.review_status,
        "approved_answer": row.approved_answer,
        "created_at": row.created_at.isoformat(sep=" ", timespec="minutes") if row.created_at else None,
        "updated_at": row.updated_at.isoformat(sep=" ", timespec="minutes") if row.updated_at else None,
    }


@router.get("/queue")
async def review_queue(
    status: str = Query("待审"),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> dict:
    rows = await review_repo.list_by_status(session, status, limit=limit)
    counts = await review_repo.counts(session)
    return {
        "status": status,
        "count": len(rows),
        "counts": counts,
        "empty": len(rows) == 0,
        "hint": EMPTY_HINT if not counts else None,
        "items": [_review_payload(r) for r in rows],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


@router.get("/trend")
async def review_trend(
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """最近几轮评估的趋势数据（旧→新）。"""
    from app.repositories import eval_run_repo

    rows = await eval_run_repo.list_eval_runs(session, limit=limit)
    items = eval_service.rows_to_trend(rows)
    return {"count": len(items), "items": items}


@router.get("/{review_id}")
async def review_detail(review_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """详情：标准化问题 + 归并进来的原话 + 当时的召回快照。

    快照是审核人判断「知识库真缺」还是「有但没检到」的唯一依据，
    所以这里连原话一起给，不让人再去翻池表。
    """
    row = await review_repo.get(session, review_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"没有 #{review_id} 这条缺口")
    sources = await low_confidence_repo.list_by_review(session, review_id)
    payload = _review_payload(row)
    payload["sources"] = [
        {
            "id": q.id,
            "raw_question": q.raw_question,
            "source": q.source,
            "reason": q.reason,
            "conversation_id": q.conversation_id,
            "created_at": q.created_at.isoformat(sep=" ", timespec="minutes") if q.created_at else None,
            "snapshot": q.retrieved_chunks or [],
        }
        for q in sources
    ]
    return payload


@router.post("/{review_id}/approve")
async def review_approve(
    review_id: int, body: ApproveRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """通过：写知识库（ch03 双写）→ 状态置通过。"""
    embedder, store = _index_assets()
    try:
        result = await flywheel.approve(
            session, review_id, body.answer, store=store, embedder=embedder
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"没有 #{review_id} 这条缺口")
    return {"ok": True, **result}


@router.post("/{review_id}/reject")
async def review_reject(review_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    row = await flywheel.reject(session, review_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"没有 #{review_id} 这条缺口")
    return {"ok": True, "id": row.id, "review_status": row.review_status}
