"""术语表与分类器单测(假 session / 假 tokenizer,不加载 400MB 权重)。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ml.topic import classifier as clf
from app.ml.topic.taxonomy import LABELS, NEIGHBOR_RULES, validate_labels


def test_taxonomy_has_17_labels_with_boundaries():
    assert len(LABELS) == 17
    assert len(set(LABELS)) == 17  # 不许重名
    assert LABELS[0] == "退换货" and LABELS[-1] == "其他"
    for label in LABELS:
        # 每类都要有一句边界说明,不能有一类是空的
        assert clf.__name__ and len(LABELS) == 17
    from app.ml.topic.taxonomy import BOUNDARIES

    assert set(BOUNDARIES) == set(LABELS)
    assert all(len(text) > 4 for text in BOUNDARIES.values())
    assert "修归保修维修" in NEIGHBOR_RULES


def test_validate_labels_dedupes_and_drops_unknown():
    assert validate_labels(["退换货", "退换货", "尺码"]) == ["退换货", "尺码"]
    assert validate_labels(["退货", "尺码"]) == ["尺码"]  # 自造类目被丢掉
    assert validate_labels([]) == ["其他"]  # 不许空标签
    assert validate_labels(["不存在的类"]) == ["其他"]


def test_contract_labels_match_taxonomy():
    assert clf.contract_labels() == list(LABELS)


class FakeTokenizer:
    """把文本长度当 token id:让测试能预测每一行的 logits 走向。"""

    def encode_batch(self, texts):
        return [
            SimpleNamespace(ids=[len(t)] * 4, attention_mask=[1] * 4, type_ids=[0] * 4)
            for t in texts
        ]


class FakeSession:
    """按「第几条」返回预设 logits,不关心输入。"""

    def __init__(self, logits: list[list[float]]):
        self._logits = logits
        self.calls = 0

    def run(self, names, inputs):
        self.calls += 1
        assert names == ["logits"]
        assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
        return [np.asarray(self._logits, dtype=np.float32)]


def _logits_for(labels_hits: dict[str, float], floor: float = -6.0) -> list[float]:
    """按 {类目: 概率} 造一行 logits,其余类目压到 floor(概率≈0)。"""
    row = [floor] * len(LABELS)
    for label, prob in labels_hits.items():
        row[LABELS.index(label)] = float(np.log(prob / (1 - prob)))
    return row


def test_predict_returns_multi_label_hits():
    session = FakeSession([_logits_for({"退换货": 0.95, "尺码": 0.66})])
    model = clf.TopicClassifier(session, FakeTokenizer(), list(LABELS), threshold=0.20)

    out = model.predict(["买大了想退"])
    assert len(out) == 1
    assert out[0].labels == ["退换货", "尺码"]
    assert out[0].scores["退换货"] == pytest.approx(0.95, abs=1e-3)
    assert session.calls == 1


def test_threshold_filters_and_can_fall_back_to_top1():
    session = FakeSession([_logits_for({"物流": 0.12})])
    model = clf.TopicClassifier(session, FakeTokenizer(), list(LABELS), threshold=0.20)

    assert model.predict(["快递"])[0].labels == []
    fallback = model.predict(["快递"], fallback_to_top1=True)[0]
    assert fallback.labels == ["物流"]
    assert fallback.scores["物流"] == pytest.approx(0.12, abs=1e-3)


def test_predict_empty_batch_short_circuits():
    session = FakeSession([])
    model = clf.TopicClassifier(session, FakeTokenizer(), list(LABELS))
    assert model.predict([]) == []
    assert session.calls == 0


def test_missing_model_dir_raises_clear_error(tmp_path):
    with pytest.raises(clf.TopicModelUnavailable) as exc:
        clf.TopicClassifier.from_model_dir(tmp_path)
    assert "fetch_topic_model" in str(exc.value)


@pytest.mark.skipif(
    not (clf.Path("models/topic_classifier") / clf.ONNX_FILE).exists(),
    reason="模型资产未拉取(scripts/fetch_topic_model.py)",
)
def test_real_model_hits_two_labels_on_multi_intent_sentence():
    """验收 3:多诉求句要同时命中多个类目。

    断言选两句各含**两个**诉求的句子。不拿「买大了想退,运费谁出」这种三诉求短句当断言:
    实测三个诉求挤在一句里,「尺码」的分数从单句 0.65 掉出前四——短句里特征互相稀释,
    这是模型的真实短板(记在 dev-notes/ch10.md),不该写成「测试必须通过」来遮住。
    """
    model = clf.load_classifier(force=True)
    cases = {
        "买大了想退": {"退换货", "尺码"},
        "退货运费谁出,优惠券还退吗": {"运费", "优惠活动"},
    }
    for text, expect in cases.items():
        labels = set(model.predict([text])[0].labels)
        assert expect <= labels, f"{text} → {sorted(labels)}"
