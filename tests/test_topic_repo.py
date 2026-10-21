"""topic_classifications 数据层单测:幂等写入、挑未归类、分布统计。"""

from __future__ import annotations

from sqlalchemy import select

from app.models.topic_classification import TopicClassification
from app.repositories import low_confidence_repo, topic_repo


async def _seed_pool(session, texts: list[str], source: str = "retrieval_low_conf") -> list[int]:
    ids = []
    for text in texts:
        row = await low_confidence_repo.add_question(session, raw_question=text, source=source)
        ids.append(row.id)
    return ids


async def test_upsert_inserts_once_then_updates(session):
    ids = await _seed_pool(session, ["买大了想退"])
    qid = ids[0]

    assert await topic_repo.upsert_many(session, [(qid, ["退换货", "尺码"])]) == 1
    rows = list((await session.execute(select(TopicClassification))).scalars().all())
    assert len(rows) == 1
    assert rows[0].labels_list == ["退换货", "尺码"]

    # 换模型重跑:同一问题只更新标签,不长新行
    assert await topic_repo.upsert_many(session, [(qid, ["尺码"])]) == 1
    rows = list((await session.execute(select(TopicClassification))).scalars().all())
    assert len(rows) == 1
    assert rows[0].labels_list == ["尺码"]


async def test_upsert_many_is_batch(session):
    ids = await _seed_pool(session, ["快递到哪了", "发票怎么开", "能便宜点吗"])
    written = await topic_repo.upsert_many(
        session, [(ids[0], ["物流"]), (ids[1], ["发票"]), (ids[2], ["优惠活动"])]
    )
    assert written == 3
    assert len(await topic_repo.list_classified_ids(session)) == 3


async def test_list_unclassified_skips_classified(session):
    ids = await _seed_pool(session, ["a 问题", "b 问题", "c 问题"])
    await topic_repo.upsert_many(session, [(ids[1], ["其他"])])

    pending = await topic_repo.list_unclassified(session)
    assert [q.id for q in pending] == [ids[0], ids[2]]

    # limit 生效且仍按入池顺序(旧的先归)
    assert [q.id for q in await topic_repo.list_unclassified(session, limit=1)] == [ids[0]]


async def test_distribution_counts_each_label(session):
    ids = await _seed_pool(session, ["买大了想退", "退货要几天", "快递到哪了"])
    await topic_repo.upsert_many(
        session,
        [
            (ids[0], ["退换货", "尺码"]),
            (ids[1], ["退换货"]),
            (ids[2], ["物流"]),
        ],
    )
    dist = await topic_repo.distribution(session)
    assert dist["退换货"] == 2
    assert dist["尺码"] == 1
    assert dist["物流"] == 1
    assert dist["发票"] == 0


async def test_list_by_label_and_latest_join_question_text(session):
    ids = await _seed_pool(session, ["买大了想退", "快递到哪了"])
    await topic_repo.upsert_many(session, [(ids[0], ["退换货", "尺码"]), (ids[1], ["物流"])])

    picked = await topic_repo.list_by_label(session, "尺码")
    assert len(picked) == 1
    question, classification = picked[0]
    assert question.raw_question == "买大了想退"
    assert classification.labels_list == ["退换货", "尺码"]

    assert {q.raw_question for q, _ in await topic_repo.latest(session)} == {
        "买大了想退",
        "快递到哪了",
    }


async def test_total_reports_classified_and_pool(session):
    ids = await _seed_pool(session, ["a", "b", "c"])
    await topic_repo.upsert_many(session, [(ids[0], ["其他"])])
    classified, pool = await topic_repo.total(session)
    assert (classified, pool) == (1, 3)


async def test_upsert_empty_list_is_noop(session):
    assert await topic_repo.upsert_many(session, []) == 0
