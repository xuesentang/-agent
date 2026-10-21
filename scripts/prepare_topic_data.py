"""数据四步（第十章需求 1）：清洗 → 术语表归并 → 80/10/10 分层 → 只增强训练集。

    # 用仓库内冻结数据 + LLM 造数，产出 v2
    python scripts/prepare_topic_data.py

    # 只用冻结数据
    python scripts/prepare_topic_data.py --no-synth

    # 指定造数文件与增强倍数
    python scripts/prepare_topic_data.py --synth data/topic_v2/synthetic-20261021-1530.jsonl --copies 2

产物落 `data/topic_v2/`（`/data/` 已 gitignore，生成物不入库），同时打一份
`reports/ch10_topic_dataset.md`——行数、每类分布、丢了哪些越界标签、被增强了几条，
这些数在面试里被问到时得张口就来。

**v2 ≠ 训练集**：已交付模型的 0.9687 来自 v1 冻结集。v2 只证明四步链路跑得通。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FROZEN_DIR = Path("app/ml/topic/data")
OUT_DIR = Path("data/topic_v2")
REPORT = Path("reports/ch10_topic_dataset.md")
VERSION = "topic-v2"


def _load_frozen() -> list[dict]:
    from app.ml.topic.dataset import read_jsonl

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        path = FROZEN_DIR / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"缺冻结数据：{path}")
        rows.extend(read_jsonl(path))
    return rows


def _latest_synth() -> Path | None:
    files = sorted(OUT_DIR.glob("synthetic-*.jsonl"))
    return files[-1] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="主题分类数据集四步处理（产出 v2）")
    parser.add_argument("--synth", default=None, help="LLM 造数 jsonl（默认取最新一份）")
    parser.add_argument("--no-synth", action="store_true", help="不并入造数")
    parser.add_argument("--copies", type=int, default=1, help="训练集每条增强几份")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from app.ml.topic.dataset import (
        augment_train,
        clean_text,
        dedupe,
        map_rows,
        primary_label,
        read_jsonl,
        stratify_split,
        write_split,
    )
    from app.ml.topic.taxonomy import LABELS

    # 0) 取料
    raw = _load_frozen()
    synth_path = None if args.no_synth else Path(args.synth) if args.synth else _latest_synth()
    if synth_path and synth_path.exists():
        synth_rows = read_jsonl(synth_path)
        raw.extend(synth_rows)
        print(f"并入造数 {len(synth_rows)} 条：{synth_path}")
    else:
        print("未并入造数（只用冻结数据）")

    # 1) 清洗
    cleaned: list[dict] = []
    changed = 0
    for row in raw:
        text = clean_text(row.get("text", ""))
        if not text:
            continue
        if text != row.get("text"):
            changed += 1
        cleaned.append({**row, "text": text, "dataset_version": VERSION})
    kept, dup = dedupe(cleaned)
    print(f"清洗：{len(raw)} → {len(kept)} 条（改写 {changed} 条，去重丢 {len(dup)} 条）")

    # 2) 术语表归并
    merged, dropped = map_rows(kept)
    print(f"归并：越界标签 {sum(dropped.values())} 个" + (f"，样本 {dict(dropped)}" if dropped else ""))

    # 3) 分层切分
    splits = stratify_split(merged, seed=args.seed)
    print(
        "切分：" + " / ".join(f"{name} {len(rows)}" for name, rows in splits.items())
        + f"（训练集占比 {len(splits['train']) / max(1, len(merged)) * 100:.1f}%）"
    )

    # 4) 只增强训练集
    train_rows, aug_stats = augment_train(splits["train"], copies=args.copies, seed=args.seed)
    print(f"增强：训练集 {len(splits['train'])} → {len(train_rows)} 条（{dict(aug_stats)}）")

    # 落盘
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "train": write_split(OUT_DIR / "train.jsonl", train_rows),
        "val": write_split(OUT_DIR / "val.jsonl", splits["val"]),
        "test": write_split(OUT_DIR / "test.jsonl", splits["test"]),
    }
    manifest = {
        "dataset_version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "v2 只证明四步链路，模型分数来自 v1 冻结集",
        "inputs": {
            "frozen": str(FROZEN_DIR),
            "synth": str(synth_path) if synth_path and synth_path.exists() else None,
        },
        "params": {"copies": args.copies, "seed": args.seed},
        "cleaning": {"input_rows": len(raw), "rewritten": changed, "duplicates_dropped": len(dup)},
        "label_merging": {"dropped_labels": dict(dropped)},
        "augmentation": dict(aug_stats),
        "files": files,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 报告：每类在三个 split 里的分布，一眼看出哪类被切空了
    dist: dict[str, Counter] = {name: Counter() for name in files}
    for name, rows in (("train", train_rows), ("val", splits["val"]), ("test", splits["test"])):
        for row in rows:
            dist[name][primary_label(row["labels"])] += 1
    lines = [
        "# CH10 数据集 v2（四步处理结果）",
        "",
        f"- 生成时间：{manifest['created_at']}",
        f"- 输入行数：{len(raw)}（含造数 {len(raw) - len(_load_frozen())} 条）",
        f"- 清洗后：{len(kept)}（改写 {changed}，去重丢 {len(dup)}）",
        f"- 越界标签：{sum(dropped.values())} 个",
        f"- 切分：train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}"
        f"（切分后、增强前；增强后训练集 {len(train_rows)}）",
        f"- 增强：仅训练集，{dict(aug_stats)}",
        "",
        "> 已交付模型的 micro-F1 0.9687 来自 **v1 冻结集**；v2 只证明四步链路跑得通。",
        "",
        "## 每类分布（按主标签，train 列含增强副本）",
        "",
        "| 类目 | train | val | test |",
        "|---|---|---|---|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {dist['train'][label]} | {dist['val'][label]} | {dist['test'][label]} |"
        )
    empty = [label for label in LABELS if not dist["val"][label] or not dist["test"][label]]
    lines += [
        "",
        f"验证/测试集为空的类目：{'、'.join(empty) if empty else '无'}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"产物：{OUT_DIR}/train.jsonl,val.jsonl,test.jsonl,manifest.json")
    print(f"报告：{REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
