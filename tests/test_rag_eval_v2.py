"""评测真实化(v2)的纯逻辑单测:自检器、加权口径、裁判回落。

这三块都是「错了会让指标失真但不报错」的地方:
- 自检器漏检 → 标签对不上库,指标低得像是检索不行;
- 加权算反 → 报告里的加权平均与算术平均倒挂;
- 裁判回落没生效 → 报告以为换了裁判,其实还是自己判自己。
"""

from __future__ import annotations

import pytest

from app.config import Settings
from tests.eval.run_rag_eval import CaseResult, _recall, _mrr, _wrecall, _wmrr
from tests.eval.validate_rag_cases import check

SECTIONS = {"退货退款政策 > 七日无理由退货", "鞋靴产品手册 > W20 入门缓震跑鞋"}


def _case(**over):
    base = {
        "id": "A01",
        "query": "多久能退货",
        "bucket": "A_policy",
        "answerable": True,
        "expected_section": "七日无理由退货",
        "expected_keywords": ["7 日内可申请退货"],
        "source": "simulated",
        "noise": [],
        "weight": 1,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- 自检器


def test_valid_case_set_passes():
    cases = [
        _case(),
        _case(
            id="D01",
            bucket="D",
            subtype="near",
            answerable=False,
            expected_section="",
            expected_keywords=[],
        ),
    ]
    assert check(cases, SECTIONS) == []


def test_unknown_section_is_an_error():
    """标签写错一个字 → 那题永远命中不了,必须报出来。"""
    errors = check([_case(expected_section="七日无理由退货规则")], SECTIONS)
    assert any("在库里找不到" in e for e in errors)


def test_refuse_case_must_not_carry_ground_truth():
    cases = [
        _case(
            id="D01",
            bucket="D",
            subtype="far",
            answerable=False,
            expected_section="七日无理由退货",
        )
    ]
    assert any("不得带 ground truth" in e for e in check(cases, SECTIONS))


def test_refuse_case_requires_subtype():
    cases = [
        _case(
            id="D01",
            bucket="D",
            answerable=False,
            expected_section="",
            expected_keywords=[],
        )
    ]
    assert any("subtype" in e for e in check(cases, SECTIONS))


def test_noise_variant_must_have_base_case():
    cases = [_case(id="A01n", noise=["typo"])]
    assert any("找不到原题" in e for e in check(cases, SECTIONS))


def test_answerable_case_requires_keywords():
    """没有要点就量不出覆盖度,等于报告里少一列指标。"""
    errors = check([_case(expected_keywords=[])], SECTIONS)
    assert any("expected_keywords" in e for e in errors)


# ---------------------------------------------------------------- 加权口径


def _results():
    # 两条:rank=1 权重 1;rank=None(未命中)权重 4。
    return [
        CaseResult("a", "A_policy", 1, weight=1),
        CaseResult("b", "A_policy", None, weight=4),
    ]


def test_arithmetic_and_weighted_diverge():
    rs = _results()
    assert _recall(rs, 5) == pytest.approx(0.5)  # 算术:2 条中 1 条命中
    assert _wrecall(rs, 5) == pytest.approx(0.2)  # 加权:5 份权重里只有 1 份命中
    assert _mrr(rs) == pytest.approx(0.5)
    assert _wmrr(rs) == pytest.approx(0.2)


def test_empty_results_are_zero_not_error():
    assert _recall([], 5) == 0.0
    assert _mrr([]) == 0.0
    assert _wrecall([], 5) == 0.0
    assert _wmrr([]) == 0.0


# ---------------------------------------------------------------- 裁判回落


def test_judge_falls_back_to_chat_model(monkeypatch):
    import app.core.llm as llm

    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: Settings(llm_model="gen-model", judge_model="", judge_base_url="", judge_api_key=""),
    )
    model, name = llm.get_judge_model()
    assert name == "gen-model"
    assert model.model_name == "gen-model"


def test_judge_uses_own_slot_when_configured(monkeypatch):
    """配了裁判就必须真的换模型名与端点,只改名不换端点等于没换。"""
    import app.core.llm as llm

    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: Settings(
            llm_model="gen-model",
            llm_base_url="https://gen.example/v1/",
            judge_model="judge-model",
            judge_base_url="https://judge.example/v1/",
            judge_api_key="judge-key",
        ),
    )
    model, name = llm.get_judge_model()
    assert name == "judge-model"
    assert model.model_name == "judge-model"
    assert "judge.example" in str(model.openai_api_base)
