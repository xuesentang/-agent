"""Eval scorer matching rules (review Critical #1 fix).

claim_type / order_id must match exactly (containment would confuse
"refund" with "return_refund"); only expected_solution uses containment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "eval"))

from run_eval import score_case  # noqa: E402
from run_tool_eval import score_case as score_tool_case  # noqa: E402


def test_exact_match_pass():
    ok, failures = score_case(
        {"order_id": "123", "claim_type": "refund", "expected_solution": "退款"},
        {"order_id": "123", "claim_type": "refund", "expected_solution": "全额退款"},
    )
    assert ok and failures == []


def test_refund_vs_return_refund_must_fail():
    """'refund' expected, 'return_refund' got -> FAIL (was silently PASS)."""
    ok, failures = score_case(
        {"order_id": None, "claim_type": "refund", "expected_solution": None},
        {"order_id": None, "claim_type": "return_refund", "expected_solution": None},
    )
    assert not ok and failures == ["claim_type"]


def test_error_response_always_fails():
    """Extraction exception yields {'error': ...}; every expected field -> FAIL."""
    ok, failures = score_case(
        {"order_id": None, "claim_type": "refund", "expected_solution": "退款"},
        {"error": "boom"},
    )
    assert not ok


def test_null_vs_null_passes_null_vs_value_fails():
    ok, _ = score_case({"order_id": None}, {"order_id": None})
    assert ok
    ok, failures = score_case({"order_id": None}, {"order_id": "编造"})
    assert not ok and failures == ["order_id"]


# ------------------------------------------------ ch02 工具选择评分器(同名不同源)
def test_tool_exact_match_pass():
    assert score_tool_case("create_ticket", "create_ticket")


def test_tool_error_prefix_always_fails_even_against_none():
    """error:* 恒 FAIL——上游报错不得蒙中标注,包括「none」。"""
    assert not score_tool_case("none", "error:boom")


def test_tool_none_matches_none_only():
    """none 只对 none 过;模型误调工具不得算蒙对。"""
    assert score_tool_case("none", "none")
    assert not score_tool_case("none", "query_faq")


def test_tool_name_must_not_be_substring_match():
    """query_order 与 query_order_status 是不同工具,子串会误判 PASS。"""
    assert not score_tool_case("query_order", "query_order_status")
