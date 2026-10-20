"""意图识别准确率 eval:35 条标注样例,七类各 5 条,走真实模型。

为什么这章需要 eval 而不是充分单测:意图判断是模型的活,单测只能验解析与兜底
(假模型喂什么返回什么)。准确率只有在真模型上跑标注样例才有意义。

用法:
    python -m tests.eval.run_intent_eval
    python -m tests.eval.run_intent_eval --out reports/ch05_intent.md

阈值 90%。失败样本逐条打印,便于直接看是 prompt 的锅还是标注的锅。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASES_PATH = Path(__file__).parent / "ch05_intent_cases.jsonl"
PASS_THRESHOLD = 0.9


async def run() -> tuple[list[dict], dict[str, list[int]]]:
    from app.agent.intent import classify_intent
    from app.core.llm import get_extract_model

    model = get_extract_model()
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict] = []
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for case in cases:
        result = await classify_intent(case["text"], model=model)
        ok = result.intent == case["expected"]
        per_class[case["expected"]][0] += int(ok)
        per_class[case["expected"]][1] += 1
        rows.append(
            {
                "text": case["text"],
                "expected": case["expected"],
                "got": result.intent,
                "confidence": result.confidence,
                "fallback": result.fallback,
                "ok": ok,
            }
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {case['text'][:26]:28s} expected={case['expected']} got={result.intent} conf={result.confidence:.2f}")
    return rows, per_class


def render(rows: list[dict], per_class: dict[str, list[int]]) -> str:
    passed = sum(1 for r in rows if r["ok"])
    lines = [
        "# CH05 意图识别 eval",
        "",
        f"用例 {len(rows)} 条(七类各 5 条),通过 {passed} 条,准确率 **{passed / len(rows):.1%}**。",
        "",
        "| 意图 | 命中/总数 | 准确率 |",
        "|---|---|---|",
    ]
    for intent, (ok, total) in per_class.items():
        lines.append(f"| {intent} | {ok}/{total} | {ok / total:.0%} |")
    fails = [r for r in rows if not r["ok"]]
    lines += ["", f"## 错误样本({len(fails)} 条)", ""]
    if not fails:
        lines.append("无。")
    for r in fails:
        lines.append(f"- 「{r['text']}」标注 {r['expected']},判成 {r['got']}(置信度 {r['confidence']:.2f},兜底={r['fallback']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="把结果写成 markdown 报告")
    args = parser.parse_args()

    rows, per_class = asyncio.run(run())
    passed = sum(1 for r in rows if r["ok"])
    rate = passed / len(rows) if rows else 0.0
    print(f"\n{passed}/{len(rows)} passed, rate={rate:.0%}, threshold={PASS_THRESHOLD:.0%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(render(rows, per_class), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0 if rate >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
