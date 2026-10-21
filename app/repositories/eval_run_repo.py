"""eval_runs / intent_usage 数据访问（ch09）。

两张表都是「只追加」的流水：评估一轮追加一行、一轮对话追加一行。
没有更新路径，所以这里只有 add + list。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval_run import TRIGGERS, EvalRun
from app.models.intent_usage import TOKEN_SOURCES, IntentUsage


async def add_eval_run(
    session: AsyncSession,
    *,
    triggered_by: str,
    dataset_size: int,
    metrics: dict,
    commit: bool = True,
) -> EvalRun:
    if triggered_by not in TRIGGERS:
        raise ValueError(f"未知触发方式：{triggered_by}")
    row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    return row


async def list_eval_runs(session: AsyncSession, limit: int = 10) -> list[EvalRun]:
    """最近 N 轮，**旧→新**返回。

    顺序是给趋势图用的：图上从左到右就是时间往后走，倒序返回会让画图的人
    每处都要记得反转一次。
    """
    stmt = select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)
    rows = list((await session.execute(stmt)).scalars().all())
    return list(reversed(rows))


async def add_usage(
    session: AsyncSession,
    *,
    intent: str,
    route: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    token_source: str = "estimated",
    conversation_id: int | None = None,
    model: str | None = None,
    commit: bool = True,
) -> IntentUsage:
    if token_source not in TOKEN_SOURCES:
        raise ValueError(f"未知 token 来源：{token_source}")
    row = IntentUsage(
        conversation_id=conversation_id,
        intent=intent or "未知",
        route=route,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        token_source=token_source,
        model=model,
    )
    session.add(row)
    if commit:
        await session.commit()
    return row


async def usage_by_intent(session: AsyncSession) -> list[dict]:
    """按意图聚合。占位与排序在这里算好，前端不再二次加工。"""
    rows = (await session.execute(select(IntentUsage))).scalars().all()
    bucket: dict[str, dict] = {}
    for row in rows:
        item = bucket.setdefault(
            row.intent,
            {
                "intent": row.intent,
                "turns": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "sources": set(),
            },
        )
        item["turns"] += 1
        item["prompt_tokens"] += int(row.prompt_tokens or 0)
        item["completion_tokens"] += int(row.completion_tokens or 0)
        item["total_tokens"] += row.total_tokens
        item["sources"].add(row.token_source)

    grand = sum(item["total_tokens"] for item in bucket.values())
    out = []
    for item in bucket.values():
        item["share"] = round(item["total_tokens"] / grand * 100, 1) if grand else 0.0
        item["avg_tokens"] = round(item["total_tokens"] / item["turns"], 1) if item["turns"] else 0.0
        item["sources"] = sorted(item["sources"])
        out.append(item)
    return sorted(out, key=lambda item: -item["total_tokens"])
