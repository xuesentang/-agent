"""置信度闸单测:两层判定的先后、fail-closed、入池原因分级。

自评那层用假模型(config 里可控),分数那层直接造 Citation —— 闸是纯函数,
不需要真向量库。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.gate import (
    REASON_DEGRADED,
    REASON_NO_HITS,
    REASON_SCORE_LOW,
    REASON_SELF_CHECK,
    check_gate,
    citations_from_payload,
)
from app.rag.citations import Citation


def _citation(score: float, n: int = 1) -> Citation:
    return Citation(
        n=n,
        chunk_id=n,
        section_path="退货退款政策 > 运费规则",
        question="退货运费谁承担",
        answer="七日无理由退货的寄回运费由消费者承担。",
        score=score,
    )


class FakeCheckModel:
    def __init__(self, content: str = '{"useful": true, "reason": "ok"}', error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content)


async def test_empty_citations_block_without_calling_model():
    model = FakeCheckModel()
    gate = await check_gate("退货运费谁出", [], min_score=0.45, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_NO_HITS
    assert model.calls == 0


async def test_low_score_blocks_before_spending_a_selfcheck_call():
    """分数挡得住的就不该花自评的钱:这是两层顺序的意义。"""
    model = FakeCheckModel()
    gate = await check_gate("帮我写一首诗", [_citation(0.39)], min_score=0.45, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_SCORE_LOW
    assert gate.top_score == 0.39
    assert model.calls == 0
    assert gate.pool_source() == "retrieval_low_conf"


async def test_high_score_still_needs_selfcheck_to_pass():
    model = FakeCheckModel('{"useful": true, "reason": "证据直接给了运费规则"}')
    gate = await check_gate("退货运费谁出", [_citation(0.72)], min_score=0.45, model=model)
    assert gate.passed is True
    assert gate.reason == ""
    assert model.calls == 1


async def test_high_score_but_selfcheck_says_no_is_blocked():
    model = FakeCheckModel('{"useful": false, "reason": "证据讲的是运费，没回答这个型号"}')
    gate = await check_gate("W40 楦宽多少毫米", [_citation(0.62)], min_score=0.45, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_SELF_CHECK
    assert "没回答这个型号" in gate.detail
    assert gate.pool_source() == "self_check"
    assert gate.degraded is False


async def test_selfcheck_failure_blocks_fail_closed():
    """CH05 的态度:自评挂了就当没判成,不放行。"""
    model = FakeCheckModel(error=RuntimeError("upstream 502"))
    gate = await check_gate("邮费多少", [_citation(0.70)], min_score=0.45, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_DEGRADED
    assert gate.degraded is True
    assert "upstream 502" in gate.detail


async def test_bad_json_also_blocks_fail_closed():
    model = FakeCheckModel("我觉得还行")
    gate = await check_gate("邮费多少", [_citation(0.70)], min_score=0.45, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_DEGRADED


def test_pool_reason_is_length_capped():
    """reason 列有长度上限,长错误信息不能把它憋成截断报错。"""
    from app.agent.gate import GateResult

    g = GateResult(False, REASON_DEGRADED, "x" * 500)
    assert len(g.pool_reason()) <= 255


def test_payload_roundtrip():
    c = _citation(0.5)
    restored = citations_from_payload([c.to_payload()])
    assert restored[0].section_path == c.section_path
    assert restored[0].score == pytest.approx(0.5, abs=1e-6)
