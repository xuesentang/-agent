"""用 ch04 评估集校准置信度闸的阈值（ch09 需求 3）。

    python scripts/calibrate_gate.py                 # 默认 hybrid_rerank、不跑 query 改写
    python scripts/calibrate_gate.py --top-k 5 --strategy hybrid   # 换策略看曲线

## 校准口径

- **正例**：评估集里 D 桶以外的问题（政策直问 / 点名型号 / 黑话 / 一句问两件事），
  这些是「库里确实有答案」的，闸把它们拦下就是误杀。
- **负例**：D 桶 14 条（far 6 一眼库外、near 8 相关但答不了），闸要拦的就是它们。
- 对每条真实检索结果算 `evidence_confidence`（与线上完全同一个函数，不另写一份），
  扫 0.20~0.95，记录每个阈值下的「正例误杀率」与「负例拦下率」。

选点规则：**正例误杀 ≤ 10%** 的前提下，负例拦下率最高者；打平取误杀更低、阈值更小
（阈值越小越保守，宁可多问一次自评）。这不是最优化问题——两个错法代价不同，
误杀一次正例=用户被拒答，漏放一次负例=答错，前者更贵，所以先卡误杀。

产物：`reports/ch09_gate_calibration.md`（曲线表 + 选定值 + 每条 case 的信号明细）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")
os.environ.setdefault("VECTOR_STORE_URI", "./data/ch07_milvus.db")

CASES = Path("tests/eval/ch04_cases.jsonl")
REPORT = Path("reports/ch09_gate_calibration.md")
MAX_FALSE_KILL = 0.10  # 正例误杀上限


def _load_cases(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


async def main_async(args) -> int:
    from app.agent.gate import evidence_confidence
    from app.config import get_settings
    from app.db import get_sessionmaker
    from app.rag.retriever import retrieve

    settings = get_settings()
    min_score = settings.kb_min_score
    cases = _load_cases(CASES)
    session_factory = get_sessionmaker()

    rows: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        question = case.get("question") or case.get("query") or ""
        hits = await retrieve(
            session_factory=session_factory,
            query=question,
            top_k=args.top_k,
            strategy=args.strategy,
            rewrite=args.rewrite,
        )
        signals = evidence_confidence(
            [type("C", (), {"score": h.score})() for h in hits], min_score=min_score
        )
        negative = case.get("bucket") == "D"
        rows.append(
            {
                "case_id": case.get("id") or f"case-{idx}",
                "bucket": case.get("bucket"),
                "negative": negative,
                "hits": len(hits),
                "top1": round(signals.top1, 4),
                "gap": round(signals.gap, 4),
                "effective": signals.effective,
                "confidence": round(signals.confidence, 4),
            }
        )
        print(f"[{idx}/{len(cases)}] {case.get('bucket'):>11} conf={signals.confidence:.3f} {question[:30]}")

    positives = [r for r in rows if not r["negative"]]
    negatives = [r for r in rows if r["negative"]]

    curve = []
    for step in range(20, 96):  # 0.20 ~ 0.95
        threshold = step / 100
        false_kill = sum(1 for r in positives if r["confidence"] < threshold) / max(1, len(positives))
        blocked = sum(1 for r in negatives if r["confidence"] < threshold) / max(1, len(negatives))
        curve.append({"threshold": threshold, "false_kill": round(false_kill, 4), "blocked": round(blocked, 4)})

    feasible = [c for c in curve if c["false_kill"] <= MAX_FALSE_KILL]
    chosen = max(feasible, key=lambda c: (c["blocked"], -c["false_kill"], -c["threshold"])) if feasible else None

    lines = [
        "# CH09 置信度闸校准",
        "",
        f"- 评估集：{CASES}（正例 {len(positives)} / 负例 {len(negatives)}）",
        f"- 策略：{args.strategy}，top_k={args.top_k}，query 改写：{'开' if args.rewrite else '关'}",
        f"- 有效证据门槛 KB_MIN_SCORE={min_score}，公式 0.60*top1 + 0.25*min(有效/3,1) + 0.15*min(分差/0.30,1)",
        f"- 选点规则：正例误杀 ≤ {MAX_FALSE_KILL:.0%} 下负例拦下率最高",
        "",
    ]
    if chosen:
        lines += [
            f"## 选定阈值：**{chosen['threshold']:.2f}**",
            "",
            f"- 正例误杀 {chosen['false_kill']:.1%}（{round(chosen['false_kill'] * len(positives))}/{len(positives)}）",
            f"- 负例拦下 {chosen['blocked']:.1%}（{round(chosen['blocked'] * len(negatives))}/{len(negatives)}）",
            "",
        ]
    else:
        lines += ["## 没有满足误杀上限的阈值——说明分数分布重叠严重，需要看明细", ""]

    lines += ["## 曲线（0.20 ~ 0.95）", "", "| 阈值 | 正例误杀 | 负例拦下 |", "|---|---|---|"]
    for c in curve:
        mark = " ←选定" if chosen and abs(c["threshold"] - chosen["threshold"]) < 1e-9 else ""
        lines.append(f"| {c['threshold']:.2f} | {c['false_kill']:.1%} | {c['blocked']:.1%}{mark} |")

    lines += ["", "## 每条 case 的信号", "", "| case | 桶 | top1 | 分差 | 有效证据 | confidence |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['case_id']} | {r['bucket']} | {r['top1']:.3f} | {r['gap']:.3f} | "
            f"{r['effective']} | {r['confidence']:.3f} |"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{REPORT}")
    if chosen:
        print(f"建议阈值：{chosen['threshold']:.2f}（误杀 {chosen['false_kill']:.1%}，拦下 {chosen['blocked']:.1%}）")
        print(f"当前配置 EVIDENCE_MIN_CONFIDENCE={settings.evidence_min_confidence}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="置信度闸阈值校准")
    parser.add_argument("--strategy", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rewrite", action="store_true", help="是否跑 query 改写（慢，要调模型）")
    args = parser.parse_args()
    import asyncio

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
