"""主题分类器评测:留出测试集上每类精确率/召回率/F1 + 混淆矩阵(第十章验收 1)。

跑的是 ONNX 推理这条真实链路(和上线用的同一个分类器),不是从实验报告里抄数字。
测试集是 finetune-lab 冻结的 150 条:训练、选阈值都没碰过它,所以这份报告可以对外。

    python scripts/eval_topic_model.py
    python scripts/eval_topic_model.py --split val          # 想看验证集也行(报告会标 split)

输出:
    reports/ch10_topic_eval.md    人看的报告(每类 P/R/F1/support + 混淆矩阵)
    reports/ch10_topic_eval.json  机读的同款数字
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "app" / "ml" / "topic" / "data"
REPORTS = ROOT / "reports"
# lab 报告里的基准,用来抓「模型资产接错 / 阈值改错」这类低级错误
BASELINE = {"test": 0.9687, "val": 0.9858}
BASELINE_TOLERANCE = 0.01


def load_rows(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"缺数据文件:{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(rows: list[dict], predictions) -> dict:
    """按类目算 P/R/F1/support 与混淆矩阵的四个格子。

    多标签口径和 sklearn 的 multilabel 一致:每类各自算二分类,再汇总 micro/macro。
    """
    from app.ml.topic.taxonomy import LABELS

    per_class = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0} for label in LABELS}
    total_hits = 0
    empty_preds = 0

    for row, pred in zip(rows, predictions):
        gold = set(row["labels"])
        got = set(pred.labels)
        if not got:
            empty_preds += 1
        total_hits += len(got)
        for label in LABELS:
            g, p = label in gold, label in got
            cell = per_class[label]
            if g:
                cell["support"] += 1
            if g and p:
                cell["tp"] += 1
            elif p and not g:
                cell["fp"] += 1
            elif g and not p:
                cell["fn"] += 1
            else:
                cell["tn"] += 1

    micro_tp = sum(c["tp"] for c in per_class.values())
    micro_fp = sum(c["fp"] for c in per_class.values())
    micro_fn = sum(c["fn"] for c in per_class.values())
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    classes = []
    f1_values = []
    for label in LABELS:
        cell = per_class[label]
        p = cell["tp"] / (cell["tp"] + cell["fp"]) if cell["tp"] + cell["fp"] else 0.0
        r = cell["tp"] / (cell["tp"] + cell["fn"]) if cell["tp"] + cell["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1_values.append(f1)
        classes.append(
            {
                "name": label,
                "p": round(p, 4),
                "r": round(r, 4),
                "f1": round(f1, 4),
                "support": cell["support"],
                **{k: cell[k] for k in ("tp", "fp", "fn", "tn")},
            }
        )

    return {
        "rows": len(rows),
        "avg_labels_per_row": round(total_hits / len(rows), 3) if rows else 0.0,
        "empty_predictions": empty_preds,
        "micro": {"p": round(micro_p, 4), "r": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "classes": classes,
    }


def render_markdown(report: dict, *, split: str, threshold: float, baseline: float | None) -> str:
    lines = [
        f"# 主题分类器评测({split} 集)",
        "",
        f"- 样本数:**{report['rows']}**",
        f"- 阈值:{threshold}",
        f"- 平均标签数/句:**{report['avg_labels_per_row']}**",
        f"- 完全没命中(空标签)的句子:{report['empty_predictions']} 条",
        f"- **micro P/R/F1**:{report['micro']['p']} / {report['micro']['r']} / **{report['micro']['f1']}**",
        f"- **macro F1**:{report['macro_f1']}",
    ]
    if baseline is not None:
        delta = report["micro"]["f1"] - baseline
        lines.append(f"- 与 lab 报告基准({baseline})对比:**{delta:+.4f}**")
    lines += [
        "",
        "## 每类指标",
        "",
        "| 类目 | 精确率 | 召回率 | F1 | support | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in report["classes"]:
        lines.append(
            f"| {cell['name']} | {cell['p']:.4f} | {cell['r']:.4f} | {cell['f1']:.4f} | "
            f"{cell['support']} | {cell['tp']} | {cell['fp']} | {cell['fn']} | {cell['tn']} |"
        )

    lines += [
        "",
        "## 每类混淆矩阵",
        "",
        "> 多标签下每类是一个 2×2:行是真值(有/无),列是预测(有/无)。",
        "",
        "| 类目 | TP(有→有) | FN(有→无) | FP(无→有) | TN(无→无) |",
        "|---|---|---|---|---|",
    ]
    for cell in report["classes"]:
        lines.append(
            f"| {cell['name']} | {cell['tp']} | {cell['fn']} | {cell['fp']} | {cell['tn']} |"
        )

    weakest = sorted(report["classes"], key=lambda c: c["f1"])[:3]
    lines += [
        "",
        "## 最弱三类",
        "",
        "| 类目 | F1 | support | 典型错法 |",
        "|---|---|---|---|",
    ]
    for cell in weakest:
        pattern = (
            f"漏判 {cell['fn']} 次 / 误判 {cell['fp']} 次"
            if cell["support"]
            else "测试集里没有这类样本(指标不可信)"
        )
        lines.append(f"| {cell['name']} | {cell['f1']:.4f} | {cell['support']} | {pattern} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="主题分类器评测")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--threshold", type=float, default=None, help="默认取配置 TOPIC_THRESHOLD")
    parser.add_argument("--out", default=str(REPORTS), help="报告输出目录")
    parser.add_argument("--no-baseline", action="store_true", help="不与 lab 报告比对")
    args = parser.parse_args()

    from app.config import get_settings
    from app.ml.topic.classifier import load_classifier

    settings = get_settings()
    threshold = args.threshold if args.threshold is not None else settings.topic_threshold
    rows = load_rows(args.split)

    try:
        model = load_classifier(force=True)
    except Exception as exc:  # noqa: BLE001 - 模型缺失时给出可执行的下一步
        print(f"加载模型失败:{exc}", file=sys.stderr)
        return 2
    model.threshold = threshold
    predictions = model.predict([row["text"] for row in rows])

    report = evaluate(rows, predictions)
    baseline = None if args.no_baseline else BASELINE.get(args.split)
    report["split"] = args.split
    report["threshold"] = threshold
    report["baseline"] = baseline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ch10_topic_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report, split=args.split, threshold=threshold, baseline=baseline)
    (out_dir / "ch10_topic_eval.md").write_text(markdown, encoding="utf-8")

    print(f"{args.split}: rows={report['rows']} micro_f1={report['micro']['f1']} "
          f"macro_f1={report['macro_f1']} threshold={threshold}")
    print(f"报告:{out_dir / 'ch10_topic_eval.md'}")

    if baseline is not None:
        delta = report["micro"]["f1"] - baseline
        if abs(delta) > BASELINE_TOLERANCE:
            print(
                f"与 lab 基准差 {delta:+.4f},超过容差 {BASELINE_TOLERANCE}:"
                "模型资产、阈值或数据被换过了,先查这三样",
                file=sys.stderr,
            )
            return 1
        print(f"与 lab 基准差 {delta:+.4f}(容差 {BASELINE_TOLERANCE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
