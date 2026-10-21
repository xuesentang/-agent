"""主题归类旁路服务（第十章需求 5）。

一句话：从低置信度问题池里挑还没归过类的问题，批量丢给分类器，结果写
`topic_classifications`。**实时对话不调用它**——需求写得很清楚，主链路不该多背一个
400MB 的模型。

## 三个"不"

- **不重算**：只挑池里没归过类的。跑十遍和跑一遍结果一样，第二遍挑到 0 条。
- **不阻塞**：分类器缺依赖/缺模型就返回 `reason`，调用方（脚本/定时任务）记一行日志跳过，
  绝不抛给用户请求。
- **不合并多个入口的数据**：池子里 `source` 有三个入口，这里不区分——归类的目的是
  「哪类问题堆得多」，谁送进来的不影响主题。
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.models.topic_classification import TopicClassification
from app.repositories import topic_repo

logger = logging.getLogger(__name__)


def _empty(reason: str) -> dict:
    return {
        "picked": 0,
        "classified": 0,
        "reason": reason,
        "under_batch": False,
        "distribution": {},
        "preview": [],
    }


async def classify_pending(
    session_factory,
    *,
    batch_size: int | None = None,
    min_batch: int = 1,
    classifier=None,
    dry_run: bool = False,
) -> dict:
    """跑一批归类。返回统计，不抛业务异常。

    参数：
    - `batch_size`：这一批最多取多少条（默认 `TOPIC_BATCH_SIZE`）；
    - `min_batch`：攒够多少条才肯跑。定时任务可以设成 50（攒够一批再算），
      手工验收设 1（有几条归几条）。不足时返回 `under_batch=True` 且不写库；
    - `classifier`：注入用（单测传假分类器）；不传就按配置加载 ONNX；
    - `dry_run`：只算不写，用来先看模型给出的标签对不对。
    """
    settings = get_settings()
    size = batch_size or settings.topic_batch_size

    if classifier is None and not settings.topic_enabled:
        return _empty("TOPIC_ENABLED=false(旁路默认关,验收时打开)")

    if classifier is None:
        from app.ml.topic.classifier import TopicModelUnavailable, load_classifier

        try:
            classifier = load_classifier()
        except TopicModelUnavailable as exc:
            logger.warning("主题归类跳过:分类器不可用(%s)", exc)
            return _empty(f"分类器不可用:{exc}")

    async with session_factory() as session:
        pending = await topic_repo.list_unclassified(session, limit=size)
        if not pending:
            return _empty("池子里没有未归类的问题")
        if len(pending) < min_batch:
            result = _empty(f"不足一批({len(pending)} < {min_batch}),等攒够再跑")
            result["picked"] = len(pending)
            result["under_batch"] = True
            return result

        predictions = classifier.predict(
            [q.raw_question for q in pending], fallback_to_top1=True
        )
        rows = [
            (question.id, list(pred.labels))
            for question, pred in zip(pending, predictions)
            if pred.labels
        ]

        if dry_run:
            return {
                "picked": len(pending),
                "classified": 0,
                "reason": "dry-run:未写库",
                "under_batch": False,
                "distribution": {},
                "preview": [
                    {
                        "question_id": question.id,
                        "text": question.raw_question,
                        "labels": pred.labels,
                        "scores": pred.scores,
                    }
                    for question, pred in zip(pending, predictions)
                ],
            }

        written = await topic_repo.upsert_many(session, rows)
        distribution = await topic_repo.distribution(session)

    logger.info(
        "主题归类完成:picked=%s written=%s 分布=%s",
        len(pending), written, dict(distribution.most_common(5)),
    )
    return {
        "picked": len(pending),
        "classified": written,
        "reason": None,
        "under_batch": False,
        "distribution": dict(distribution.most_common()),
        "preview": [],
    }


def summarize_predictions(predictions) -> dict:
    """把一批预测压成一个便于打印的分布（脚本/日志用，不写库）。"""
    counts: dict[str, int] = {}
    for pred in predictions:
        for label in pred.labels:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def label_of(classification: TopicClassification) -> list[str]:
    """归类行的标签列表（薄封装，避免调用方到处 json.loads）。"""
    return classification.labels_list
