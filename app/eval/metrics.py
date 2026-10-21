"""检索评估指标（从 `tests/eval/run_rag_eval.py` 抽出来的公共部分，ch09 D7）。

抽出来的理由：评估脚本里那套 R@5 / R@10 / MRR 口径要开始被别的地方复用
（定时流水线写 `eval_runs`、后台页画趋势），留在测试脚本里就只能 import 测试代码。
算法一行没改，脚本改成 import 之后数值必须与 `reports/ch04_eval_v2.md` 一致——
这条等价性有单测钉着。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
K5, K10 = 5, 10
TOP10_SHARE_WARN = 0.05  # Top-10 占全库超过 5% 就没有区分度


@dataclass
class CaseResult:
    case_id: str
    bucket: str
    rank: int | None  # 第一条命中块的排名(1-based),None = 前 K 条都没命中
    weight: int = 1
    noise: bool = False
    hits: list = field(default_factory=list)


def load_cases(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in rows:
        r["weight"] = int(r.get("weight") or 1)
    return rows


def rank_of(hits, expected_section: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        path = getattr(hit.chunk, "section_path", "") or ""
        if expected_section and expected_section in path:
            return i
    return None


def recall(results: list[CaseResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.rank is not None and r.rank <= k) / len(results)


def mrr(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(1.0 / r.rank for r in results if r.rank is not None) / len(results)


def weighted_recall(results: list[CaseResult], k: int) -> float:
    total = sum(r.weight for r in results)
    if not total:
        return 0.0
    return sum(r.weight for r in results if r.rank is not None and r.rank <= k) / total


def weighted_mrr(results: list[CaseResult]) -> float:
    total = sum(r.weight for r in results)
    if not total:
        return 0.0
    return sum(r.weight / r.rank for r in results if r.rank is not None) / total


def summarize(results: list[CaseResult], strategy: str) -> dict:
    """一条策略的结果 → 写进 `eval_runs.metrics` 的那份字典。

    只放数字与规模，不放逐条明细：趋势图要的是能比较的几个量，
    明细在报告 markdown 里，塞进 JSON 列只会让表越来越大。
    """
    clean = [r for r in results if not r.noise]
    return {
        "strategy": strategy,
        "cases": len(results),
        "clean_cases": len(clean),
        "r5": round(recall(results, K5), 4),
        "r10": round(recall(results, K10), 4),
        "mrr": round(mrr(results), 4),
        "wr5": round(weighted_recall(results, K5), 4),
        "wr10": round(weighted_recall(results, K10), 4),
        "wmrr": round(weighted_mrr(results), 4),
        "clean_mrr": round(mrr(clean), 4),
    }
