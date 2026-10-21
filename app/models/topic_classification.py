"""topic_classifications 表:微调主题分类器的批量归类结果(第十章需求 5)。

## 为什么只有旁路写它

实时对话不调分类器。低置信度问题攒够一批才归一次类,结果落这里,喂给后台看
「哪类问题堆得多」,决定先补哪块知识。主链路多一次 400MB 模型的推理,换来的只是
一个标签——不值。

## 唯一键为什么是 question_id

一个问题只归一次类,重复跑就更新标签、不新增行。攒批脚本靠这个键幂等:
中断重跑、手滑跑两遍,表里都不会长出一堆重复行。
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt


class TopicClassification(Base):
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    # 指向 low_confidence_questions.id;一个问题一行(唯一)
    question_id: Mapped[int] = mapped_column(
        BigInt, ForeignKey("low_confidence_questions.id"), unique=True, index=True
    )
    # 多标签,17 类权威类目里命中的若干个;JSON 文本存储,读写都过 labels_list
    labels: Mapped[str] = mapped_column(Text)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    @property
    def labels_list(self) -> list[str]:
        try:
            value = json.loads(self.labels or "[]")
        except (TypeError, ValueError):
            return []
        return [str(x) for x in value] if isinstance(value, list) else []

    @staticmethod
    def dump_labels(labels: list[str]) -> str:
        return json.dumps(list(labels), ensure_ascii=False)
