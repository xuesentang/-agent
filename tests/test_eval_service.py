"""CH09 评估流水线：指标汇总、落 eval_runs、趋势顺序。

指标算法本身的等价性由 `test_rag_eval_v2.py` 钉着（那份测试现在还从脚本 import，
底层已换成 `app/eval/metrics.py`）——这里只管「跑一轮会不会正确落表」。
"""

from __future__ import annotations

import json

import pytest

from app.eval.metrics import CaseResult, summarize
from app.repositories import eval_run_repo
from app.services import eval_service


class _Chunk:
    def __init__(self, section_path: str):
        self.section_path = section_path


class _Hit:
    def __init__(self, section_path: str, score: float = 0.8):
        self.chunk = _Chunk(section_path)
        self.score = score


def _cases_file(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


CASES = [
    {"id": "A1", "bucket": "A_policy", "query": "退货运费谁出", "expected_section": "退货政策", "answerable": True},
    {"id": "A2", "bucket": "A_policy", "query": "能开发票吗", "expected_section": "发票", "answerable": True, "weight": 3},
    {"id": "D1", "bucket": "D", "query": "你们老板是谁", "answerable": False},
]


# ------------------------------------------------------------------ 指标汇总


def test_summarize_reports_both_plain_and_weighted():
    results = [
        CaseResult("A1", "A_policy", 1),
        CaseResult("A2", "A_policy", 3, weight=3),
        CaseResult("A3", "A_policy", None),
        CaseResult("A4", "A_policy", 2, noise=True),
    ]
    m = summarize(results, "hybrid_rerank")
    assert m["cases"] == 4
    assert m["clean_cases"] == 3
    assert m["r5"] == pytest.approx(0.75)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4, abs=1e-4)
    assert m["wr5"] == pytest.approx(5 / 6, abs=1e-4)  # 权重 1+3+1 命中，漏掉的那条权重 1
    assert m["clean_mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-4)
    assert m["strategy"] == "hybrid_rerank"


def test_summarize_handles_empty_results():
    m = summarize([], "dense")
    assert m["r5"] == 0.0 and m["mrr"] == 0.0 and m["cases"] == 0


# ------------------------------------------------------------------ 跑一轮


@pytest.mark.asyncio
async def test_run_round_writes_eval_run(session, session_factory, monkeypatch, tmp_path):
    async def fake_retrieve(*, session_factory, query, top_k=10, strategy="dense", retriever=None, **kw):
        if "运费" in query:
            return [_Hit("退货政策 > 运费规则"), _Hit("其它")]
        return [_Hit("其它"), _Hit("发票 > 抬头")]  # 第二名命中

    monkeypatch.setattr("app.rag.retriever.retrieve", fake_retrieve)

    result = await eval_service.run_round(
        session_factory,
        cases_path=_cases_file(tmp_path, CASES),
        retriever=object(),
        strategy="hybrid_rerank",
    )

    assert result["dataset_size"] == 2  # D 桶不可答，不进检索评估
    m = result["metrics"]
    assert m["r5"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 0.5) / 2)
    assert m["with_generation"] is False

    rows = await eval_run_repo.list_eval_runs(session)
    assert len(rows) == 1
    assert rows[0].triggered_by == "手动"
    assert rows[0].dataset_size == 2
    assert rows[0].metrics["strategy"] == "hybrid_rerank"


@pytest.mark.asyncio
async def test_run_round_honours_limit(session_factory, monkeypatch, tmp_path):
    async def fake_retrieve(*, session_factory, query, top_k=10, strategy="dense", retriever=None, **kw):
        return [_Hit("退货政策 > 运费规则")]

    monkeypatch.setattr("app.rag.retriever.retrieve", fake_retrieve)
    result = await eval_service.run_round(
        session_factory, cases_path=_cases_file(tmp_path, CASES), retriever=object(), limit=1
    )
    assert result["dataset_size"] == 1


@pytest.mark.asyncio
async def test_run_round_rejects_unknown_trigger(session_factory, monkeypatch, tmp_path):
    async def fake_retrieve(**kw):
        return []

    monkeypatch.setattr("app.rag.retriever.retrieve", fake_retrieve)
    with pytest.raises(ValueError):
        await eval_service.run_round(
            session_factory,
            cases_path=_cases_file(tmp_path, CASES),
            retriever=object(),
            triggered_by="谁按的",
        )


# ------------------------------------------------------------------ 趋势


@pytest.mark.asyncio
async def test_trend_returns_oldest_first(session, session_factory):
    for i in range(3):
        await eval_run_repo.add_eval_run(
            session,
            triggered_by="手动",
            dataset_size=10 + i,
            metrics={"strategy": "hybrid_rerank", "r5": 0.5 + i * 0.01, "mrr": 0.4, "with_generation": False},
        )

    rows = await eval_service.trend(session_factory, limit=10)

    assert [r["dataset_size"] for r in rows] == [10, 11, 12]  # 旧 → 新
    assert rows[-1]["r5"] == pytest.approx(0.52)
    assert rows[0]["created_at"] is not None


@pytest.mark.asyncio
async def test_trend_respects_limit(session, session_factory):
    for i in range(5):
        await eval_run_repo.add_eval_run(
            session, triggered_by="定时", dataset_size=i, metrics={"r5": 0.1 * i}
        )
    rows = await eval_service.trend(session_factory, limit=2)
    assert [r["dataset_size"] for r in rows] == [3, 4]
