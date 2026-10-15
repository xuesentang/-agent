"""Run the labeled after-sales extraction eval against the real LLM API.

Requires a valid LLM_API_KEY in .env (Task 9 prerequisite).
Exit code 0 iff pass rate >= 90%.

Usage:
    python -m tests.eval.run_eval
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

CASES_PATH = Path(__file__).parent / "after_sales_cases.jsonl"
PASS_THRESHOLD = 0.9
FIELDS = ("order_id", "claim_type", "expected_solution")


EXACT_FIELDS = ("order_id", "claim_type")  # containment would conflate refund/return_refund
CONTAINMENT_FIELDS = ("expected_solution",)


def _match_field(expected: Any, got: Any, exact: bool) -> bool:
    """Null must match null. Exact fields compare by ==; expected_solution by containment."""
    if expected is None:
        return got is None
    if got is None:
        return False
    return str(expected) == str(got) if exact else str(expected) in str(got)


def score_case(expected: dict, got: dict) -> tuple[bool, list[str]]:
    if "error" in got:
        return False, list(FIELDS)  # extraction exception: always FAIL, never a null-hit pass
    failures = [
        field
        for field in FIELDS
        if not _match_field(
            expected.get(field), got.get(field), exact=field in EXACT_FIELDS
        )
    ]
    return not failures, failures


async def run(extract: Callable | None = None) -> int:
    if extract is None:
        from app.services.extract import extract_after_sales as extract

    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for i, case in enumerate(cases, 1):
        try:
            got = await extract(case["text"])
        except Exception as exc:  # noqa: BLE001 - eval reports, never crashes
            got = {"error": str(exc)[:120]}
        ok, failures = score_case(case["expected"], got)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] #{i} {case['text'][:28]}... failures={failures} got={got}")
    rate = passed / len(cases) if cases else 0.0
    print(f"\n{passed}/{len(cases)} passed, rate={rate:.0%}, threshold={PASS_THRESHOLD:.0%}")
    return 0 if rate >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
