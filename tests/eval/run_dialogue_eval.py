"""多轮对话 eval:指代消解 + 意图识别,走真实模型(第六章验收 1、2)。

单测只能验解析层(喂什么返回什么),「每一轮意图都判对、指代都补全对」只有拿
标注样例在真模型上跑才有意义。这个脚本复刻图里的前两个节点:
`understand_query`(带历史)→ `classify_intent`(判补全后的问题,同样带历史)。

历史怎么造:用户消息用**原话**,客服回复用一句不含信息量的占位话术。这样指代
只能靠用户自己说过的话补全,不给模型额外提示——真链路上客服回复里当然会有单号,
拿它做提示等于降低难度,测出来的数字不作数。

用法:
    python -m tests.eval.run_dialogue_eval
    python -m tests.eval.run_dialogue_eval --out reports/ch06_dialogue.md

三条指标,任一不达标即失败:
- 意图准确率 ≥ 90%(含单轮边界题,如「你觉得我该买哪双」这类**不该**进兜底的)
- 指代补全准确率 ≥ 90%(只统计标了 resolved_must_contain 的轮次)
- 怪问题落「其他」= 全中(验收 2 明确要求拿不准的归兜底)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASES_PATH = Path(__file__).parent / "ch06_dialogue_cases.jsonl"
INTENT_THRESHOLD = 0.9
RESOLVE_THRESHOLD = 0.9
PLACEHOLDER_REPLY = "好的,已经帮您处理了。"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_scenario(case: dict, *, understand_model, intent_model) -> list[dict]:
    """跑一个多轮场景,每轮都带上到目前为止的历史。"""
    from app.agent.intent import classify_intent
    from app.agent.query_understanding import understand_query

    history: list[dict] = []
    rows: list[dict] = []
    for turn in case["turns"]:
        text = turn["text"]
        understood = await understand_query(text, history=history, model=understand_model)
        result = await classify_intent(understood.resolved, model=intent_model, history=history)

        must = turn.get("resolved_must_contain") or []
        resolve_ok = all(token in understood.resolved for token in must) if must else None
        intent_ok = result.intent == turn["intent"]

        rows.append(
            {
                "scenario": case["id"],
                "desc": case.get("desc", ""),
                "text": text,
                "resolved": understood.resolved,
                "canonical": understood.canonical,
                "order_id": understood.order_id,
                "expected_intent": turn["intent"],
                "got_intent": result.intent,
                "confidence": result.confidence,
                "fallback": result.fallback,
                "must": must,
                "intent_ok": intent_ok,
                "resolve_ok": resolve_ok,
            }
        )
        print(
            f"[{'PASS' if intent_ok else 'FAIL'}] {case['id']} {text[:20]:22s} "
            f"intent={result.intent}(期望 {turn['intent']},conf={result.confidence:.2f})"
            + (f" resolve={resolve_ok}" if resolve_ok is not None else "")
        )
        if understood.resolved != text:
            print(f"        ↳ 补全:{understood.resolved}")

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": PLACEHOLDER_REPLY})
    return rows


async def run_odd(case: dict, *, intent_model) -> dict:
    """单轮怪问题 / 边界题:不经历史,单独判一次。

    期望「其他」的进怪问题组(验兜底);期望别的(比如「你觉得我该买哪双」→ 商品咨询)
    是边界题,验的是**不要滥用兜底**,归到意图准确率里算。
    """
    from app.agent.intent import classify_intent

    result = await classify_intent(case["text"], model=intent_model)
    ok = result.intent == case["intent"]
    print(f"[{'PASS' if ok else 'FAIL'}] {case['id']} {case['text'][:20]:22s} intent={result.intent}(期望 {case['intent']})")
    return {
        "id": case["id"],
        "text": case["text"],
        "expected": case["intent"],
        "got": result.intent,
        "confidence": result.confidence,
        "fallback": result.fallback,
        "ok": ok,
    }


async def run() -> dict:
    from app.core.llm import get_extract_model

    # 两个模型槽位分开取:理解层和意图层用的都是"结构化输出"那一路
    understand_model = get_extract_model()
    intent_model = get_extract_model()

    cases = load_cases()
    turns: list[dict] = []
    for case in cases:
        if case.get("odd"):
            continue
        turns.extend(await run_scenario(case, understand_model=understand_model, intent_model=intent_model))

    singles = [await run_odd(c, intent_model=intent_model) for c in cases if c.get("odd")]
    odds = [r for r in singles if r["expected"] == "其他"]
    bounds = [r for r in singles if r["expected"] != "其他"]
    for r in bounds:
        turns.append(
            {
                "scenario": r["id"],
                "desc": "单轮边界题",
                "text": r["text"],
                "resolved": r["text"],
                "canonical": r["text"],
                "order_id": None,
                "expected_intent": r["expected"],
                "got_intent": r["got"],
                "confidence": r["confidence"],
                "fallback": r["fallback"],
                "must": [],
                "intent_ok": r["ok"],
                "resolve_ok": None,
            }
        )

    intent_total = len(turns)
    intent_pass = sum(1 for r in turns if r["intent_ok"])
    resolve_rows = [r for r in turns if r["resolve_ok"] is not None]
    resolve_pass = sum(1 for r in resolve_rows if r["resolve_ok"])
    odd_pass = sum(1 for r in odds if r["ok"])
    return {
        "turns": turns,
        "odds": odds,
        "bounds": bounds,
        "intent": (intent_pass, intent_total),
        "resolve": (resolve_pass, len(resolve_rows)),
        "odd": (odd_pass, len(odds)),
    }


def render(data: dict) -> str:
    turns, odds = data["turns"], data["odds"]
    ip, it = data["intent"]
    rp, rt = data["resolve"]
    op, ot = data["odd"]
    lines = [
        "# CH06 多轮对话 eval",
        "",
        f"- 多轮轮次 **{it}** 轮,意图判对 **{ip}/{it} = {ip / it:.1%}**" if it else "- 多轮轮次 0",
        f"- 指代补全 **{rp}/{rt} = {rp / rt:.1%}**" if rt else "- 指代补全 0 条用例",
        f"- 怪问题落「其他」**{op}/{ot} = {op / ot:.1%}**" if ot else "- 怪问题 0 条",
        "",
        "## 多轮明细",
        "",
        "| 场景 | 用户原话 | 补全后 | 期望意图 | 判定 | 置信度 |",
        "|---|---|---|---|---|---|",
    ]
    for r in turns:
        mark = "PASS" if r["intent_ok"] else "**FAIL**"
        target = r["expected_intent"] + ("" if r["resolve_ok"] is None else f"(指代 {'OK' if r['resolve_ok'] else 'FAIL'})")
        lines.append(
            f"| {r['scenario']} | {r['text']} | {r['resolved']} | {target} | {mark} | {r['confidence']:.2f} |"
        )

    lines += ["", "## 怪问题", "", "| 用例 | 原话 | 期望 | 实得 | 置信度 | JSON 兜底 |", "|---|---|---|---|---|---|"]
    for r in odds:
        lines.append(
            f"| {r['id']} | {r['text']} | {r['expected']} | {r['got']} | {r['confidence']:.2f} | "
            f"{'是' if r['fallback'] else '否'} |"
        )

    fails = [r for r in turns if not r["intent_ok"]] + [r for r in turns if r["resolve_ok"] is False]
    lines += ["", f"## 失败样本({len(fails)} 条)", ""]
    if not fails:
        lines.append("无。")
    for r in fails:
        lines.append(f"- {r['scenario']}「{r['text']}」→ 补全「{r['resolved']}」,判成 {r['got_intent']}(期望 {r['expected_intent']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="把结果写成 markdown 报告")
    args = parser.parse_args()

    data = asyncio.run(run())
    ip, it = data["intent"]
    rp, rt = data["resolve"]
    op, ot = data["odd"]
    intent_rate = ip / it if it else 1.0
    resolve_rate = rp / rt if rt else 1.0
    print(
        f"\n意图 {ip}/{it} = {intent_rate:.0%}(阈值 {INTENT_THRESHOLD:.0%})  "
        f"指代 {rp}/{rt} = {resolve_rate:.0%}(阈值 {RESOLVE_THRESHOLD:.0%})  "
        f"怪问题 {op}/{ot}"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(render(data), encoding="utf-8")
        print(f"report -> {args.out}")
    ok = intent_rate >= INTENT_THRESHOLD and resolve_rate >= RESOLVE_THRESHOLD and op == ot
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
