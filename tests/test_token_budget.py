"""上下文预算纯函数单测:token 口径、倒推公式、预算不足告警。"""

from __future__ import annotations

import pytest

from app.core.token_budget import (
    EVIDENCE_TOKENS_PER_CHUNK,
    PER_STEP_OUTPUT_TOKENS,
    SUMMARY_RESERVED_TOKENS,
    compute_budget,
    estimate_tokens,
)


def test_chinese_counts_one_token_per_char():
    assert estimate_tokens("你好") == 2
    # 「订单 1001 的物流到哪了」= 汉字 8 + 空格与数字 6 → 8 + ceil(6/4) = 10
    assert estimate_tokens("订单 1001 的物流到哪了") == 10


def test_ascii_counts_four_chars_per_token():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # 向上取整
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_is_way_below_langchain_counter():
    """LangChain 的近似计数按英文 BPE 数中文,报五倍;这里必须显著更低。"""
    from langchain_core.messages.utils import count_tokens_approximately

    text = "你是电商平台官方售后客服。语气礼貌、简洁、专业。"
    assert estimate_tokens(text) < count_tokens_approximately(text) / 2


def test_budget_subtracts_every_fixed_cost():
    b = compute_budget(
        window=18000,
        system_tokens=1000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    evidence = 5 * EVIDENCE_TOKENS_PER_CHUNK
    react = 3 * (1200 + PER_STEP_OUTPUT_TOKENS)
    safety = int(18000 * 0.05)
    expected = 18000 - (1000 + evidence + SUMMARY_RESERVED_TOKENS + 2000 + 2000) - react - safety
    assert b.history_total == expected
    assert b.reserved_evidence == evidence
    assert b.reserved_react_peak == react
    assert b.reserved_summary == SUMMARY_RESERVED_TOKENS


def test_layers_split_seventy_thirty():
    b = compute_budget(
        window=18000,
        system_tokens=1000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    assert b.layer1 == int(b.history_total * 0.7)
    assert b.layer1 + b.layer2 == b.history_total
    assert b.fits is True


def test_too_small_window_warns_and_zeroes_history(caplog):
    with caplog.at_level("WARNING"):
        b = compute_budget(
            window=4000,
            system_tokens=1000,
            max_output_tokens=2000,
            max_user_input_tokens=2000,
            max_agent_steps=3,
            tool_result_max_tokens=1200,
            rerank_top_k=5,
        )
    assert b.history_total == 0
    assert b.fits is False
    assert b.layer1 == 0 and b.layer2 == 0
    assert "上下文预算不足" in caplog.text


def test_more_rerank_chunks_means_less_history():
    base = dict(
        window=18000,
        system_tokens=1000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
    )
    a = compute_budget(rerank_top_k=5, **base)
    b = compute_budget(rerank_top_k=10, **base)
    assert a.history_total - b.history_total == 5 * EVIDENCE_TOKENS_PER_CHUNK


def test_dense_only_when_rerank_off():
    b = compute_budget(
        window=18000,
        system_tokens=1000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=0,
    )
    assert b.reserved_evidence == 0


def test_tool_schemas_count_into_fixed_cost():
    """工具定义是每轮不变的前缀,必须算进固定开销(需求第 3 点)。"""
    base = dict(
        window=18000,
        system_tokens=1000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    without = compute_budget(tool_tokens=0, **base)
    with_tools = compute_budget(tool_tokens=400, **base)
    assert without.history_total - with_tools.history_total == 400
    assert with_tools.reserved_system == 1400


def test_default_tool_schemas_are_countable():
    from app.core.token_budget import tool_schema_tokens

    tokens = tool_schema_tokens()
    assert 200 < tokens < 800, f"工具定义 token 估算异常:{tokens}"


def test_demo_config_reproduces_layer_split():
    """演示配置下的层 1 / 层 2 预算(验收 2 引用的那组数)。

    与需求原文的 5650 / 3954 / 1695 差 10 token 以内,差在 system prompt 与工具
    定义的实测字数上(需求那组按它自己的提示词量)。算式结构一致:layer1 取 7 成向下取整,
    layer2 = 总量 - layer1(不各取三成,否则两个数加起来会比总量少 1-2)。
    """
    from app.core.prompts import build_agent_system_prompt
    from app.core.token_budget import tool_schema_tokens

    b = compute_budget(
        window=18000,
        system_tokens=estimate_tokens(build_agent_system_prompt(max_steps=3)),
        tool_tokens=tool_schema_tokens(),
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    assert abs(b.history_total - 5650) <= 20
    assert abs(b.layer1 - 3954) <= 15
    assert abs(b.layer2 - 1695) <= 15
