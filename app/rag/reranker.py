"""重排器(ch04):bge-reranker-v2-m3 精排。

粗排(向量/BM25/混合)只保证「大致相关」,重排用交叉编码器把 query 与每条候选
拼成一个输入对逐条打分,精度高但代价是 O(候选数) 次前向——所以只对粗排的
Top-N 做,不全库跑。

模型懒加载 + 进程内单例:2G 级权重不在 import 期加载,启动预热统一在 lifespan。
测试用 FakeReranker(词重叠打分),不依赖模型文件。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# 交叉编码器单对输入长度上限(bge-reranker-v2-m3 支持 8192,512 足够覆盖一个块)
MAX_PAIR_LENGTH = 512


class Reranker(Protocol):
    """打分接口:给 query 与若干候选文本,返回每条的相关性分(越大越相关)。"""

    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def _bigrams(text: str) -> set[str]:
    out: set[str] = set()
    for piece in _TOKEN_RE.findall((text or "").lower()):
        if piece.isascii():
            out.add(piece)
        else:
            out.update(piece[i : i + 2] for i in range(max(1, len(piece) - 1)))
    return out


class FakeReranker:
    """测试替身:按 query 与候选的 2-gram 覆盖率打分(0~1),纯确定性。"""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        q = _bigrams(query)
        if not q:
            return [0.0 for _ in docs]
        return [len(q & _bigrams(d)) / len(q) for d in docs]


class LocalBGEReranker:
    """sentence-transformers CrossEncoder 包一层,懒加载。"""

    def __init__(self, model_path: str | None = None) -> None:
        self._path = model_path
        self._model = None

    def _ensure(self):
        if self._model is None:
            from app.config import get_settings

            from sentence_transformers import CrossEncoder

            path = self._path or get_settings().rerank_model or DEFAULT_RERANK_MODEL
            logger.info("加载重排模型:%s", path)
            self._model = CrossEncoder(path, max_length=MAX_PAIR_LENGTH)
        return self._model

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        model = self._ensure()
        scores = model.predict([(query, d) for d in docs])
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_default_reranker() -> Reranker:
    """进程内单例:真 bge-reranker-v2-m3(配置项 RERANK_MODEL 可指本地目录)。"""
    return LocalBGEReranker()
