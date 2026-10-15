"""Eval scorer matching rules (review Critical #1 fix).

claim_type / order_id must match exactly (containment would confuse
"refund" with "return_refund"); only expected_solution uses containment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "eval"))

from run_eval import score_case  # noqa: E402


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
