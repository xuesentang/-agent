"""主题分类器训练（第十章需求 3）。全参微调 RoBERTa-wwm-ext，不用 LoRA。

来源：`D:\\python-kefu-agent\\finetune-lab\\train.py`，搬进仓库时改了三处——

1. **标签从 `app/ml/topic/taxonomy.py` 取**，不再抄一份常量：训练用的位次必须和推理、
   预标 Prompt、前端图表完全同源，抄四份迟早有一份先过期。
2. **路径改成仓库约定**：数据在 `app/ml/topic/data/`，基座与产物在 `models/`（已 gitignore）。
3. 测试集评测只跑一次：本脚本最后会用**验证集选出的阈值**跑测试集并落
   `final_report.json`；日常看指标用 `scripts/eval_topic_model.py`（ONNX 链路，秒级），
   别反复跑这里。

用法：

    python scripts/train_topic_model.py benchmark --samples 500   # 先摸耗时与内存
    python scripts/train_topic_model.py full --train-samples 1500 --epochs 8

基座不在 `models/chinese-roberta-wwm-ext` 时先拉一次：
    python scripts/fetch_topic_model.py --with-base
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "app" / "ml" / "topic" / "data"
BASE_MODEL = ROOT / "models" / "chinese-roberta-wwm-ext"
RUNS = ROOT / "models" / "topic_classifier" / "run"

from app.ml.topic.taxonomy import LABELS  # noqa: E402  - 上面要先改 sys.path

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
SEED = 42
BASE_NAME = "hfl/chinese-roberta-wwm-ext"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _require_training_deps():
    """训练依赖只在真跑训练时 import：只看代码、跑单测的人不需要装 torch。"""
    try:
        import numpy as np
        import psutil  # noqa: F401
        import torch
        from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_recall_fscore_support  # noqa: F401
        from torch.optim import AdamW  # noqa: F401
        from torch.utils.data import DataLoader, Dataset  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise SystemExit(
            "训练依赖缺失，先装: pip install torch transformers scikit-learn psutil accelerate"
        ) from exc
    return np, torch


class TopicDataset:
    """按 LABELS 位次生成 17 维 0/1 目标（多标签）。"""

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        import torch

        self.rows = rows
        self.encodings = tokenizer(
            [row["text"] for row in rows], truncation=True, max_length=max_length, padding=False
        )
        self._torch = torch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        labels = self._torch.zeros(len(LABELS), dtype=self._torch.float)
        for label in self.rows[index]["labels"]:
            labels[LABEL2ID[label]] = 1.0
        item["labels"] = labels
        return item


def make_loader(rows: list[dict], tokenizer, max_length: int, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = TopicDataset(rows, tokenizer, max_length)
    import torch

    def collate(features: list[dict]) -> dict:
        labels = torch.stack([item.pop("labels") for item in features])
        batch = tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=False, collate_fn=collate,
    )


def train_one_epoch(model, loader, optimizer, device: str) -> dict:
    import psutil
    import torch

    model.train()
    total_loss = 0.0
    peak_rss = 0
    started = time.perf_counter()
    batches = 0
    process = psutil.Process(os.getpid())
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 正则化之一:梯度裁剪
        optimizer.step()
        total_loss += float(output.loss.detach())
        batches += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
    seconds = time.perf_counter() - started
    return {
        "samples": len(loader.dataset),
        "batches": batches,
        "seconds": seconds,
        "avg_batch_seconds": seconds / max(batches, 1),
        "mean_loss": total_loss / max(batches, 1),
        "peak_rss_gb": peak_rss / 1024**3,
    }


def predict(model, loader, device: str):
    import numpy as np
    import torch

    model.eval()
    logits_all: list = []
    labels_all: list = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits_all.append(model(**batch).logits.cpu().numpy())
            labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def metrics_from_probs(probs, gold, threshold: float) -> dict:
    from sklearn.metrics import f1_score

    preds = (probs >= threshold).astype(int)
    return {
        "micro_f1": float(f1_score(gold, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold, preds, average="macro", zero_division=0)),
    }


def _load_base(path: Path, num_labels: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not path.exists():
        raise SystemExit(
            f"基座不在 {path}。先跑: python scripts/fetch_topic_model.py --with-base"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
    )
    return tokenizer, model


def benchmark(args) -> None:
    np, torch = _require_training_deps()
    from torch.optim import AdamW

    set_seed()
    tokenizer, model = _load_base(Path(args.base_model), len(LABELS))
    rows = load_jsonl(DATA / "train.jsonl")
    rows = random.Random(SEED).sample(rows, min(args.samples, len(rows)))
    loader = make_loader(rows, tokenizer, args.max_length, args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    result = train_one_epoch(model, loader, optimizer, "cpu")
    result.update({
        "kind": "benchmark",
        "samples_requested": args.samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "model": BASE_NAME,
    })
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def full_train(args) -> None:
    np, torch = _require_training_deps()
    from torch.optim import AdamW

    set_seed()
    tokenizer, model = _load_base(Path(args.base_model), len(LABELS))
    train_rows = load_jsonl(DATA / "train.jsonl")
    if args.train_samples and args.train_samples < len(train_rows):
        train_rows = random.Random(SEED).sample(train_rows, args.train_samples)
    val_rows = load_jsonl(DATA / "val.jsonl")
    test_rows = load_jsonl(DATA / "test.jsonl")

    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_rows, tokenizer, args.max_length, args.batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    history = []
    best_f1, best_epoch, best_state, patience = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        result = train_one_epoch(model, train_loader, optimizer, "cpu")
        logits, gold = predict(model, val_loader, "cpu")
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        val_05 = metrics_from_probs(probs, gold, 0.5)
        epoch_threshold, epoch_val = 0.5, val_05
        # 阈值只在验证集上扫;测试集一次都不碰
        for candidate in np.arange(0.10, 0.71, 0.05):
            candidate_metrics = metrics_from_probs(probs, gold, float(candidate))
            if candidate_metrics["micro_f1"] > epoch_val["micro_f1"]:
                epoch_threshold = round(float(candidate), 2)
                epoch_val = candidate_metrics
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in result.items()},
            "val_threshold": epoch_threshold,
            **{f"val_{k}": v for k, v in epoch_val.items()},
            "val_micro_f1_at_0_5": val_05["micro_f1"],
            "val_max_prob": float(probs.max()),
            "val_positive_at_0_5": int((probs >= 0.5).sum()),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch_val["micro_f1"] > best_f1:
            best_f1, best_epoch = epoch_val["micro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:  # 早停:连续 N 轮不涨就停
                print(f"早停:连续 {args.patience} 轮验证集没涨", flush=True)
                break
    if best_state is None:
        raise SystemExit("训练没产出可用权重")
    model.load_state_dict(best_state)

    logits, gold = predict(model, val_loader, "cpu")
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    best_threshold, best_threshold_f1, scan = 0.5, -1.0, []
    for candidate in np.arange(0.10, 0.71, 0.05):
        score = metrics_from_probs(probs, gold, float(candidate))
        scan.append({"threshold": round(float(candidate), 2), **score})
        if score["micro_f1"] > best_threshold_f1:
            best_threshold_f1 = score["micro_f1"]
            best_threshold = round(float(candidate), 2)

    from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support

    best_dir = RUNS / "best_model"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    (RUNS / "threshold.json").write_text(
        json.dumps({"threshold": best_threshold, "val_micro_f1": best_threshold_f1},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 测试集只在这里跑一次
    test_loader = make_loader(test_rows, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_logits, test_gold = predict(model, test_loader, "cpu")
    test_probs = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    test_preds = (test_probs >= best_threshold).astype(int)
    p, r, f1, support = precision_recall_fscore_support(test_gold, test_preds, zero_division=0)
    micro = metrics_from_probs(test_probs, test_gold, best_threshold)
    cms = multilabel_confusion_matrix(test_gold, test_preds)
    report = {
        "model": BASE_NAME,
        "train_size": len(train_rows), "val_size": len(val_rows), "test_size": len(test_rows),
        "best_epoch": best_epoch, "best_val_f1_at_0_5": best_f1, "threshold": best_threshold,
        "val_micro_f1_at_threshold": best_threshold_f1,
        "test_micro_f1": micro["micro_f1"], "test_macro_f1": micro["macro_f1"],
        "classes": [
            {"name": label, "p": float(p[i]), "r": float(r[i]), "f1": float(f1[i]),
             "support": int(support[i]), "tn": int(cms[i][0][0]), "fp": int(cms[i][0][1]),
             "fn": int(cms[i][1][0]), "tp": int(cms[i][1][1])}
            for i, label in enumerate(LABELS)
        ],
        "threshold_scan": scan, "history": history,
    }
    (RUNS / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {"test_micro_f1": micro["micro_f1"], "test_macro_f1": micro["macro_f1"],
         "best_threshold": best_threshold}, ensure_ascii=False), flush=True)
    print(f"记住:阈值 {best_threshold} 要同步到 .env 的 TOPIC_THRESHOLD", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="主题分类器训练")
    parser.add_argument("mode", choices=["benchmark", "full"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--base-model", default=str(BASE_MODEL))
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark(args)
    else:
        full_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
