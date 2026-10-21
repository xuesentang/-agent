"""主题分类推理服务(第十章需求 5)。

## 只给旁路用

低置信度问题攒够一批,归一次类,结果落 `topic_classifications`。实时对话不调它:
400MB 的模型挂在主链路上,换来的只是一个标签,不值。

## 依赖缺失不许拖垮服务

`onnxruntime` / `tokenizers` 在函数内 import,模型文件不在也一样——统一抛
`TopicModelUnavailable`,调用方(服务层/脚本)捕获后记一行日志跳过。旁路功能装不上
依赖不该让客服服务起不来。

## 为什么阈值不是 0.5

17 类多标签,sigmoid 后的分数普遍偏低;0.20 是验证集上扫出来的最优点
(`finetune-lab/experiments/threshold.json`)。换模型必须重新扫,所以它进配置项
`TOPIC_THRESHOLD`,不写死在代码里。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.topic.taxonomy import LABELS, validate_labels

logger = logging.getLogger(__name__)

MAX_LENGTH = 64
LABELS_PATH = Path(__file__).with_name("labels.json")
ONNX_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"


class TopicModelUnavailable(RuntimeError):
    """模型资产或推理依赖缺失。旁路功能,调用方捕获后跳过即可。"""


@dataclass(frozen=True)
class TopicPrediction:
    """一条文本的归类结果。`scores` 只留命中项,节省存储与传输。"""

    text: str
    labels: list[str]
    scores: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {"text": self.text, "labels": self.labels, "scores": self.scores}


def contract_labels() -> list[str]:
    """模型契约里的标签顺序(与 ONNX 输出位次一一对应)。

    顺序必须和 `taxonomy.LABELS` 完全一致:差一位就是全表标错,而且不会报错。
    所以这里对齐一次,不一致直接抛。
    """
    try:
        payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        labels = [str(x) for x in payload["labels"]]
    except (OSError, KeyError, ValueError) as exc:  # pragma: no cover - 文件在仓库里
        raise TopicModelUnavailable(f"读不到标签契约 {LABELS_PATH}:{exc}") from exc
    if labels != list(LABELS):
        raise TopicModelUnavailable(
            "labels.json 与 taxonomy.LABELS 顺序不一致,拒绝启动分类器(顺序错=全表标错)"
        )
    return labels


class TopicClassifier:
    """ONNX 推理。session/tokenizer 可注入,单测不必加载 400MB 权重。"""

    def __init__(self, session, tokenizer, labels: list[str], *, threshold: float = 0.20) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self.labels = list(labels)
        self.threshold = float(threshold)

    @classmethod
    def from_model_dir(
        cls, model_dir: str | Path, *, threshold: float = 0.20
    ) -> "TopicClassifier":
        base = Path(model_dir)
        onnx_path = base / ONNX_FILE
        tokenizer_path = base / TOKENIZER_FILE
        if not onnx_path.exists() or not tokenizer_path.exists():
            raise TopicModelUnavailable(
                f"主题分类模型不在 {base}(需要 {ONNX_FILE} 与 {TOKENIZER_FILE})。"
                "跑 `python scripts/fetch_topic_model.py` 从 finetune-lab 拉一次。"
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise TopicModelUnavailable(
                "缺推理依赖 onnxruntime / tokenizers,跳过主题分类"
            ) from exc

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=MAX_LENGTH)
        tokenizer.enable_padding(length=MAX_LENGTH)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        return cls(session, tokenizer, contract_labels(), threshold=threshold)

    def predict(
        self, texts: list[str], *, fallback_to_top1: bool = False
    ) -> list[TopicPrediction]:
        """批量归类。

        `fallback_to_top1=True` 时,一个标签都没过阈值的句子取概率最高那类——
        入库场景用(分布页上不能出现「无主题」黑洞);评测不用(会虚高召回)。
        """
        if not texts:
            return []
        import numpy as np

        encodings = self._tokenizer.encode_batch([str(t) for t in texts])
        inputs = {
            "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.asarray(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
            "token_type_ids": np.asarray([e.type_ids for e in encodings], dtype=np.int64),
        }
        logits = self._session.run(["logits"], inputs)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))

        results: list[TopicPrediction] = []
        for text, row in zip(texts, probs):
            hits = {
                self.labels[i]: round(float(score), 4)
                for i, score in enumerate(row)
                if float(score) >= self.threshold
            }
            if not hits and fallback_to_top1:
                top = int(np.argmax(row))
                hits = {self.labels[top]: round(float(row[top]), 4)}
            labels = validate_labels(list(hits.keys())) if hits else []
            results.append(TopicPrediction(text=str(text), labels=labels, scores=hits))
        return results


_classifier: TopicClassifier | None = None
_classifier_error: str | None = None


def load_classifier(*, force: bool = False) -> TopicClassifier:
    """进程级单例:加载一次 400MB 权重就够贵了。

    失败也缓存错误信息,避免每批都去碰一次磁盘。
    """
    global _classifier, _classifier_error
    if _classifier is not None and not force:
        return _classifier
    from app.config import get_settings

    settings = get_settings()
    try:
        _classifier = TopicClassifier.from_model_dir(
            settings.topic_model_dir, threshold=settings.topic_threshold
        )
        _classifier_error = None
    except TopicModelUnavailable as exc:
        _classifier_error = str(exc)
        raise
    return _classifier


def last_error() -> str | None:
    """最近一次加载失败的原因(后台页/脚本用来解释「为什么没有数据」)。"""
    return _classifier_error


def available() -> bool:
    """本进程里分类器是否已经加载成功。

    不触发加载:后台页问「模型在不在」不该顺手把 400MB 读进来。想看真实可用性,
    跑 `scripts/classify_topics.py` 或评测脚本,那里会真的加载一次。
    """
    return _classifier is not None
