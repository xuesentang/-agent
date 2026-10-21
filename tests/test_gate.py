"""置信度闸单测:两层判定的先后、fail-closed、入池原因分级、三信号合成。

自评那层用假模型(config 里可控),分数那层直接造 Citation —— 闸是纯函数,
不需要真向量库。

ch09 起闸有两种口径:
- 传了 `min_confidence` → 比三信号合成的 confidence(图里走这条);
- 没传 → 退化成 ch04 的「只比 top1」,给老调用方留路。
两组测试分开写,免得改一边把另一边弄坏了还看不出来。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.gate import (
    CONF_EFFECTIVE_TARGET,
    CONF_WEIGHT_COUNT,
    CONF_WEIGHT_GAP,
    CONF_WEIGHT_TOP1,
    REASON_CONFIDENCE_LOW,
    REASON_DEGRADED,
    REASON_NO_HITS,
    REASON_SCORE_LOW,
    REASON_SELF_CHECK,
    check_gate,
    citations_from_payload,
    evidence_confidence,
)
from app.rag.citations import Citation

MIN_SCORE = 0.45  # 有效证据门槛,与 settings.kb_min_score 默认值同值


def _citation(score: float, n: int = 1) -> Citation:
    return Citation(
        n=n,
        chunk_id=n,
        section_path="退货退款政策 > 运费规则",
        question="退货运费谁承担",
        answer="七日无理由退货的寄回运费由消费者承担。",
        score=score,
    )


def _citations(*scores: float) -> list[Citation]:
    """按给定分数顺序造引用,顺序即检索排名。"""
    return [_citation(s, n=i + 1) for i, s in enumerate(scores)]


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


# --------------------------------------------------------------------------
# 一、三信号合成(纯函数,不碰模型)
# --------------------------------------------------------------------------


def test_confidence_of_nothing_is_zero():
    """零命中不该靠任何信号拿分;此时闸会先按 no_hits 拦下,走不到这里。"""
    signals = evidence_confidence([], min_score=MIN_SCORE)
    assert signals.confidence == 0.0
    assert signals.effective == 0


def test_confidence_of_single_hit_gets_no_gap_penalty():
    """只有一条时 top2 记 0,分差算它拉得开 —— 孤证也是证据,不能反被扣分。"""
    signals = evidence_confidence(_citations(0.90), min_score=MIN_SCORE)
    assert signals.top2 == 0.0
    assert signals.gap == pytest.approx(0.90)
    # 分差那一项拿满,但「证据条数」只算 1 条,所以仍拿不到满分
    expected = (
        CONF_WEIGHT_TOP1 * 0.90 + CONF_WEIGHT_COUNT * (1 / CONF_EFFECTIVE_TARGET) + CONF_WEIGHT_GAP
    )
    assert signals.confidence == pytest.approx(expected)


def test_confidence_ignores_negative_gap():
    """乱序输入(后一条反而更高)时分差记 0,不出现负贡献把分数往下拽。"""
    signals = evidence_confidence(_citations(0.50, 0.70), min_score=MIN_SCORE)
    assert signals.gap == 0.0
    expected = CONF_WEIGHT_TOP1 * 0.50 + CONF_WEIGHT_COUNT * (2 / CONF_EFFECTIVE_TARGET)
    assert signals.confidence == pytest.approx(expected)


def test_confidence_count_term_saturates_at_three():
    """有效证据到 3 条就够,再多不加分 —— 否则堆引用能刷分。"""
    three = evidence_confidence(_citations(0.60, 0.60, 0.60), min_score=MIN_SCORE)
    five = evidence_confidence(_citations(0.60, 0.60, 0.60, 0.60, 0.60), min_score=MIN_SCORE)
    assert three.effective == 3
    assert five.effective == 5
    assert three.confidence == pytest.approx(five.confidence)


def test_confidence_only_counts_hits_above_the_evidence_floor():
    """低于 KB_MIN_SCORE 的条目不算一条有效证据,但照样参与分差。"""
    signals = evidence_confidence(_citations(0.80, 0.20), min_score=MIN_SCORE)
    assert signals.effective == 1
    assert signals.gap == pytest.approx(0.60)


def test_confidence_stays_within_unit_interval():
    """脏数据(分数大于 1)也不能让 confidence 出界。"""
    signals = evidence_confidence(_citations(1.50), min_score=MIN_SCORE)
    assert 0.0 <= signals.confidence <= 1.0


# --------------------------------------------------------------------------
# 二、新口径:比 confidence
# --------------------------------------------------------------------------


async def test_confidence_below_threshold_blocks_without_calling_model():
    """分数挡得住的就不该花自评的钱:这是两层顺序的意义。"""
    model = FakeCheckModel()
    gate = await check_gate(
        "帮我写一首诗",
        _citations(0.30),
        min_score=MIN_SCORE,
        min_confidence=0.75,
        model=model,
    )
    assert gate.passed is False
    assert gate.reason == REASON_CONFIDENCE_LOW
    assert gate.top_score == pytest.approx(0.30)
    assert gate.confidence < 0.75
    assert model.calls == 0
    assert gate.pool_source() == "retrieval_low_conf"


async def test_confidence_just_below_threshold_blocks():
    """边界:差一点也是不够,别让「接近阈值」变成放行理由。"""
    signals = evidence_confidence(_citations(0.60), min_score=MIN_SCORE)
    gate = await check_gate(
        "退货运费谁出",
        _citations(0.60),
        min_score=MIN_SCORE,
        min_confidence=signals.confidence + 0.01,
        model=FakeCheckModel(),
    )
    assert gate.passed is False
    assert gate.reason == REASON_CONFIDENCE_LOW


async def test_confidence_passing_still_needs_selfcheck():
    """过了粗筛不等于放行:第二层照花一次调用。"""
    model = FakeCheckModel('{"useful": true, "reason": "证据直接给了运费规则"}')
    signals = evidence_confidence(_citations(0.78, 0.76, 0.70), min_score=MIN_SCORE)
    gate = await check_gate(
        "退货运费谁出",
        _citations(0.78, 0.76, 0.70),
        min_score=MIN_SCORE,
        min_confidence=signals.confidence - 0.01,
        model=model,
    )
    assert gate.passed is True
    assert gate.reason == ""
    assert model.calls == 1


async def test_confidence_passing_but_selfcheck_says_no_is_blocked():
    model = FakeCheckModel('{"useful": false, "reason": "证据讲的是运费，没回答这个型号"}')
    gate = await check_gate(
        "W40 楦宽多少毫米",
        _citations(0.62),
        min_score=MIN_SCORE,
        min_confidence=0.50,
        model=model,
    )
    assert gate.passed is False
    assert gate.reason == REASON_SELF_CHECK
    assert "没回答这个型号" in gate.detail
    assert gate.pool_source() == "self_check"
    assert gate.degraded is False


async def test_confidence_detail_explains_which_signal_dragged_it_down():
    """入池后要能复盘「为什么判弱」,detail 里三个信号都得留下。"""
    gate = await check_gate(
        "这个牌子的鞋偏码吗",
        _citations(0.38, 0.37, 0.36),
        min_score=MIN_SCORE,
        min_confidence=0.75,
        model=FakeCheckModel(),
    )
    assert "top1=" in gate.detail
    assert "gap=" in gate.detail
    assert "有效=" in gate.detail
    assert gate.pool_reason().startswith(REASON_CONFIDENCE_LOW)


async def test_empty_citations_block_without_calling_model():
    """零命中优先于低分:没有证据和证据太弱,入池原因得分得开。"""
    model = FakeCheckModel()
    gate = await check_gate("退货运费谁出", [], min_score=MIN_SCORE, min_confidence=0.50, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_NO_HITS
    assert model.calls == 0


async def test_selfcheck_failure_blocks_fail_closed():
    """CH05 的态度:自评挂了就当没判成,不放行。"""
    model = FakeCheckModel(error=RuntimeError("upstream 502"))
    gate = await check_gate(
        "邮费多少", _citations(0.70), min_score=MIN_SCORE, min_confidence=0.50, model=model
    )
    assert gate.passed is False
    assert gate.reason == REASON_DEGRADED
    assert gate.degraded is True
    assert "upstream 502" in gate.detail


async def test_bad_json_also_blocks_fail_closed():
    model = FakeCheckModel("我觉得还行")
    gate = await check_gate(
        "邮费多少", _citations(0.70), min_score=MIN_SCORE, min_confidence=0.50, model=model
    )
    assert gate.passed is False
    assert gate.reason == REASON_DEGRADED


# --------------------------------------------------------------------------
# 三、旧口径:min_confidence 不传时退回只比 top1(ch04 行为,兼容用)
# --------------------------------------------------------------------------


async def test_without_min_confidence_it_compares_top1_only():
    model = FakeCheckModel()
    gate = await check_gate("帮我写一首诗", _citations(0.39), min_score=MIN_SCORE, model=model)
    assert gate.passed is False
    assert gate.reason == REASON_SCORE_LOW
    assert gate.top_score == pytest.approx(0.39)
    assert model.calls == 0
    assert gate.pool_source() == "retrieval_low_conf"


async def test_without_min_confidence_a_high_top1_goes_to_selfcheck():
    model = FakeCheckModel('{"useful": true, "reason": "ok"}')
    gate = await check_gate("退货运费谁出", _citations(0.72), min_score=MIN_SCORE, model=model)
    assert gate.passed is True
    assert model.calls == 1
    # 旧口径也顺手把 confidence 带上,便于对比两套阈值
    assert gate.confidence > 0


# --------------------------------------------------------------------------
# 四、杂项
# --------------------------------------------------------------------------


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
