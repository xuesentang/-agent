"""topic_classifications 数据访问(第十章)。

设计要点:

- **幂等**:`upsert_many` 按 question_id 更新标签,重复跑不长新行;
- **攒批**:`list_unclassified` 只挑还没归过类的,跑多少次都不会重算全表;
- **分布统计在 Python 侧展开**:labels 是 JSON 数组,SQLite 没有 json_each 的通用写法,
  为了单测与 MySQL 两边一致,统一把标签读回来用 Counter 数。池子是几千条量级,
  一次全捞也没问题;真到十万级再改成物化统计表。

所有函数都接受外部 AsyncSession,事务由调用方管(脚本里一段一批)。
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.low_confidence_question import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification


async def upsert_many(
    session: AsyncSession, rows: list[tuple[int, list[str]]]
) -> int:
    """批量写归类结果。返回写入(新增+更新)的条数。

    同一 (question_id) 已存在就更新标签与时间——标签可能因为换模型而变,
    但一个问题只占一行。
    """
    if not rows:
        return 0
    ids = [qid for qid, _ in rows]
    existing = {
        row.question_id: row
        for row in (
            await session.execute(
                select(TopicClassification).where(TopicClassification.question_id.in_(ids))
            )
        )
        .scalars()
        .all()
    }
    for question_id, labels in rows:
        payload = TopicClassification.dump_labels(labels)
        row = existing.get(question_id)
        if row is None:
            session.add(
                TopicClassification(question_id=question_id, labels=payload)
            )
        else:
            row.labels = payload
    await session.commit()
    return len(rows)


async def list_classified_ids(session: AsyncSession) -> set[int]:
    """已经归过类的问题 id 集合(脚本用它排重,省一次 join)。"""
    ids = (await session.execute(select(TopicClassification.question_id))).scalars().all()
    return set(ids)


async def list_unclassified(
    session: AsyncSession, limit: int | None = None
) -> list[LowConfidenceQuestion]:
    """池子里还没归类的问题,按入池先后(旧的先归,免得老问题永远排不上)。"""
    done = select(TopicClassification.question_id)
    stmt = (
        select(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.id.not_in(done))
        .order_by(LowConfidenceQuestion.id)
    )
    if limit:
        stmt = stmt.limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def list_all(
    session: AsyncSession, limit: int | None = None
) -> list[LowConfidenceQuestion]:
    """池里全部问题，按入池先后。

    预标脚本用它：预标不是归类，不挑「没归过类的」——已经归过类的问题也要预标一遍，
    才能拿 LLM 的标签和模型标签对比着看。
    """
    stmt = select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id)
    if limit:
        stmt = stmt.limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def distribution(session: AsyncSession) -> Counter:
    """各类目命中的问题数。多标签问题会在每个命中类目各计一次。"""
    labels = (
        await session.execute(select(TopicClassification.labels))
    ).scalars().all()
    counter: Counter = Counter()
    for raw in labels:
        row = TopicClassification(labels=raw)
        for label in row.labels_list:
            counter[label] += 1
    return counter


async def list_by_label(
    session: AsyncSession, label: str, limit: int = 50
) -> list[tuple[LowConfidenceQuestion, TopicClassification]]:
    """某个类目下的问题(带归类行),给后台详情列表用。"""
    stmt = (
        select(LowConfidenceQuestion, TopicClassification)
        .join(TopicClassification, TopicClassification.question_id == LowConfidenceQuestion.id)
        .order_by(TopicClassification.classified_at.desc(), TopicClassification.id.desc())
    )
    rows = (await session.execute(stmt)).all()
    # JSON 里带中文,LIKE 在 MySQL/SQLite 都能匹配(不区分大小写无所谓)
    matched = [
        (q, c) for q, c in rows if label in c.labels_list
    ][:limit]
    return matched


async def latest(
    session: AsyncSession, limit: int = 50
) -> list[tuple[LowConfidenceQuestion, TopicClassification]]:
    """最近归类的若干条。"""
    stmt = (
        select(LowConfidenceQuestion, TopicClassification)
        .join(TopicClassification, TopicClassification.question_id == LowConfidenceQuestion.id)
        .order_by(TopicClassification.classified_at.desc(), TopicClassification.id.desc())
        .limit(limit)
    )
    return [(q, c) for q, c in (await session.execute(stmt)).all()]


async def total(session: AsyncSession) -> tuple[int, int]:
    """(已归类问题数, 池内问题总数)。分布页头部两个数字。"""
    classified = (
        await session.execute(select(TopicClassification.id))
    ).scalars().all()
    pool = (await session.execute(select(LowConfidenceQuestion.id))).scalars().all()
    return len(classified), len(pool)
