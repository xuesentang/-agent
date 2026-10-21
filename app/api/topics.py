"""主题分布只读接口（第十章需求 5 的后台侧）。

三个约束：

- **只读**：这一章只做「看分布」，不从这里触发归类。要跑归类走
  `scripts/classify_topics.py`（定时任务/手工都行），后台页不该有写入口。
- **不加载模型**：接口只读数据库，不 import onnxruntime。否则后台页一开就背 400MB。
  「模型可用吗」用 `app.ml.topic.classifier.last_error()` 判断，那是懒加载的产物，
  从不主动触发加载。
- **空结果照常返回 200**：还没跑过归类时分布是空的，前端显示「攒够一批再来看」，
  不该给个 404 让人以为接口挂了。
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.config import get_settings
from app.db import get_session
from app.models.low_confidence_question import LowConfidenceQuestion
from app.repositories import topic_repo

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/topics", tags=["topics"])

UNCLASSIFIED_HINT = "还没有归类结果，先跑 python scripts/classify_topics.py"


def _model_status() -> dict:
    """分类器的就绪状态。不触发加载，只看上一次加载的结论。"""
    from app.ml.topic.classifier import available, last_error

    settings = get_settings()
    return {
        "enabled": settings.topic_enabled,
        "model_dir": settings.topic_model_dir,
        "threshold": settings.topic_threshold,
        "available": available(),
        "error": last_error(),
    }


@router.get("/distribution")
async def topic_distribution(session: AsyncSession = Depends(get_session)) -> dict:
    """各类目命中数 + 池子规模。

    `distinct` 和 `pool_total` 一起给：池子是「低置信度问题」表，同一条问题可能因为
    多次测试重复入池，39 行可能只有 13 个不同的问题。只报 39 会让人误判样本量。
    """
    counter = await topic_repo.distribution(session)
    classified, pool_total = await topic_repo.total(session)
    distinct = (
        await session.execute(
            select(func.count(func.distinct(LowConfidenceQuestion.raw_question)))
        )
    ).scalar() or 0

    from app.ml.topic.taxonomy import LABELS

    # 17 类全给出来（含 0），否则条形图会随数据长短变形，看着像「类目变少了」
    items = [
        {
            "label": label,
            "count": counter.get(label, 0),
            "share": round(counter.get(label, 0) / classified * 100, 1) if classified else 0.0,
        }
        for label in LABELS
    ]
    return {
        "total": classified,
        "pool_total": pool_total,
        "distinct": int(distinct),
        "empty": classified == 0,
        "hint": UNCLASSIFIED_HINT if classified == 0 else None,
        "labels": items,
        "model": _model_status(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


@router.get("/latest")
async def topic_latest(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """最近归类的问题（原话 + 标签 + 时间）。"""
    rows = await topic_repo.latest(session, limit=limit)
    return {
        "count": len(rows),
        "items": [
            {
                "question_id": question.id,
                "text": question.raw_question,
                "labels": classification.labels_list,
                "classified_at": classification.classified_at.isoformat()
                if classification.classified_at
                else None,
            }
            for question, classification in rows
        ],
    }


@router.get("/by-label")
async def topic_by_label(
    label: str = Query(min_length=1),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """点某个类目看它下面有哪些问题。类目名不合法时返回空列表，不报 500。"""
    from app.ml.topic.taxonomy import LABELS

    if label not in LABELS:
        return {"label": label, "count": 0, "items": [], "error": f"未知类目：{label}"}
    rows = await topic_repo.list_by_label(session, label, limit=limit)
    return {
        "label": label,
        "count": len(rows),
        "items": [
            {
                "question_id": question.id,
                "text": question.raw_question,
                "labels": classification.labels_list,
                "classified_at": classification.classified_at.isoformat()
                if classification.classified_at
                else None,
            }
            for question, classification in rows
        ],
    }
