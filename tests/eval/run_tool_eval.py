"""Run the ch02 tool-selection eval against the real LLM API.

问句 → bind_tools invoke → 模型选的工具名 vs 标注;工具名精确匹配。
「none」= 模型不调工具直接回答(闲聊/职责类问句)。
Requires a valid LLM_API_KEY in .env. Exit code 0 iff pass rate >= 90%.

Usage:
    python -m tests.eval.run_tool_eval
"""

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

CASES_PATH = Path(__file__).parent / "ch02_tool_cases.jsonl"
PASS_THRESHOLD = 0.9


def score_case(expected: str, got: str) -> bool:
    """工具名精确匹配;error:* 恒 FAIL(CH01 评分器同语义:
    错误响应不得蒙中任何标注,包括 none)。"""
    return not got.startswith("error:") and got == expected


async def select_tool(text: str) -> str:
    """真实链路的一跳:全量工具 bind 后让模型定工具,返回工具名或 none。

    必须带 system prompt 复刻真实链路——「何时调 create_ticket」等行为指引
    就写在 system prompt 里,裸发 user 消息测的不是同一条链路。
    """
    from app.core.llm import get_chat_model
    from app.core.prompts import build_chat_prompt_with_tools
    from app.core.tools.business_tools import build_default_registry

    bound = get_chat_model().bind_tools(build_default_registry().for_bind())
    prompt_messages = build_chat_prompt_with_tools().format_messages(messages=[], input=text)
    resp = await bound.ainvoke(prompt_messages)
    calls = resp.tool_calls
    return calls[0]["name"] if calls else "none"


async def run(select: Callable[[str], str] | None = None) -> int:
    if select is None:
        select = select_tool
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for i, case in enumerate(cases, 1):
        try:
            got = await select(case["text"])
        except Exception as exc:  # noqa: BLE001 - eval reports, never crashes
            got = f"error:{exc}"
        ok = score_case(case["expected"], got)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] #{i} {case['text'][:24]}... expected={case['expected']} got={got}")
    rate = passed / len(cases) if cases else 0.0
    print(f"\n{passed}/{len(cases)} passed, rate={rate:.0%}, threshold={PASS_THRESHOLD:.0%}")
    return 0 if rate >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
