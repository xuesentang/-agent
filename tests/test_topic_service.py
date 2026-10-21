"""批量归类服务单测:只挑未归类、幂等、开关与降级、dry-run 不写库。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.ml.topic.classifier import TopicModelUnavailable, TopicPrediction
from app.models.topic_classification import TopicClassification
from app.repositories import low_confidence_repo
from app.services import topic_service


class FakeClassifier:
    """按写死的映射给标签,记录调用次数——验的是编排,不是模型。"""

    def __init__(self, mapping: dict[str, list[str]] | None = None, default=("其他",)):
        self.mapping = mapping or {}
        self.default = list(default)
        self.calls: list[list[str]] = []

    def predict(self, texts, fallback_to_top1=False):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            labels = self.mapping.get(text, self.default)
            out.append(
                TopicPrediction(text=text, labels=list(labels), scores={label: 0.9 for label in labels})
            )
        return out


async def _seed(session, texts):
    ids = []
    for text in texts:
        row = await low_confidence_repo.add_question(
            session, raw_question=text, source="retrieval_low_conf"
        )
        ids.append(row.id)
    return ids


@pytest.fixture
def enabled(monkeypatch):
    """把开关打开,免得每条用例都去动 .env。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "topic_enabled", True)
    monkeypatch.setattr(settings, "topic_batch_size", 50)
    return settings


async def test_classifies_pending_and_reports_distribution(session, session_factory, enabled):
    ids = await _seed(session, ["买大了想退", "快递到哪了"])
    classifier = FakeClassifier({"买大了想退": ["退换货", "尺码"], "快递到哪了": ["物流"]})

    result = await topic_service.classify_pending(session_factory, classifier=classifier)

    assert result["picked"] == 2 and result["classified"] == 2
    assert result["distribution"]["退换货"] == 1
    assert result["distribution"]["尺码"] == 1
    rows = list((await session.execute(select(TopicClassification))).scalars().all())
    assert {row.question_id for row in rows} == set(ids)


async def test_second_run_picks_nothing(session, session_factory, enabled):
    await _seed(session, ["买大了想退"])
    classifier = FakeClassifier()
    await topic_service.classify_pending(session_factory, classifier=classifier)

    again = await topic_service.classify_pending(session_factory, classifier=classifier)
    assert again["picked"] == 0
    assert again["reason"] == "池子里没有未归类的问题"
    assert len(classifier.calls) == 1, "第二次不该再调模型"


async def test_relabel_updates_same_row(session, session_factory, enabled):
    """换模型重跑:同一个问题只更新标签,不长新行。"""
    ids = await _seed(session, ["买大了想退"])
    await topic_service.classify_pending(
        session_factory, classifier=FakeClassifier({"买大了想退": ["退换货", "尺码"]})
    )
    await topic_service.classify_pending(
        session_factory, classifier=FakeClassifier({"买大了想退": ["尺码"]})
    )
    # 已经归过类就不会再被挑中,所以这里直接验 upsert 的语义
    from app.repositories import topic_repo

    await topic_repo.upsert_many(session, [(ids[0], ["尺码"])])
    rows = list((await session.execute(select(TopicClassification))).scalars().all())
    assert len(rows) == 1 and rows[0].labels_list == ["尺码"]


async def test_min_batch_waits_for_enough(session, session_factory, enabled):
    await _seed(session, ["a", "b"])
    classifier = FakeClassifier()
    result = await topic_service.classify_pending(
        session_factory, classifier=classifier, min_batch=10
    )
    assert result["under_batch"] is True
    assert result["classified"] == 0
    assert not (await session.execute(select(TopicClassification))).scalars().all()


async def test_disabled_switch_short_circuits(session_factory, monkeypatch):
    """开关关着就什么都不做;注入了分类器算「我知道自己在干什么」,不受开关影响。"""
    monkeypatch.setattr(get_settings(), "topic_enabled", False)
    result = await topic_service.classify_pending(session_factory)
    assert result["picked"] == 0
    assert "TOPIC_ENABLED" in result["reason"]


async def test_missing_model_degrades_instead_of_raising(session_factory, enabled, monkeypatch):
    import app.ml.topic.classifier as classifier_module

    def boom(*args, **kwargs):
        raise TopicModelUnavailable("模型不在")

    monkeypatch.setattr(classifier_module, "load_classifier", boom)
    result = await topic_service.classify_pending(session_factory)
    assert result["picked"] == 0
    assert "分类器不可用" in result["reason"]


async def test_dry_run_writes_nothing(session, session_factory, enabled):
    await _seed(session, ["买大了想退"])
    result = await topic_service.classify_pending(
        session_factory, classifier=FakeClassifier(), dry_run=True
    )
    assert result["classified"] == 0
    assert result["preview"][0]["text"] == "买大了想退"
    assert not (await session.execute(select(TopicClassification))).scalars().all()


def test_summarize_predictions_counts_labels():
    preds = [
        SimpleNamespace(labels=["退换货", "尺码"]),
        SimpleNamespace(labels=["退换货"]),
    ]
    assert topic_service.summarize_predictions(preds) == {"退换货": 2, "尺码": 1}
