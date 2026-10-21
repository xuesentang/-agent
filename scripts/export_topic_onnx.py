"""把训练好的权重导出成 ONNX，并与 PyTorch 对齐（第十章需求 5 的部署前置）。

来源：`finetune-lab/export_onnx.py`，改了两处：产物直接落到推理服务读的位置
（`models/topic_classifier/model.onnx` + `tokenizer.json`），以及把「一致性校验」的
结论写进 `export_report.json`——导出成功不等于接得对，得拿 150 条测试集逐条比
logits 与阈值判定。

    python scripts/export_topic_onnx.py
    python scripts/export_topic_onnx.py --run-dir models/topic_classifier/run

导出后跑一次 `python scripts/eval_topic_model.py`：ONNX 链路的 micro-F1 必须和
PyTorch 报告一致，差一点都说明导错了。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "app" / "ml" / "topic" / "data"
OUT_DIR = ROOT / "models" / "topic_classifier"
MAX_LENGTH = 64


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="导出主题分类 ONNX")
    parser.add_argument("--run-dir", default=str(OUT_DIR / "run"), help="train_topic_model.py 的产物目录")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="推理服务读模型的位置")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    best = run_dir / "best_model"
    if not (best / "config.json").exists():
        print(f"没有可导出的权重:{best}(先跑 scripts/train_topic_model.py full)", file=sys.stderr)
        return 2
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"导出依赖缺失:{exc}", file=sys.stderr)
        return 2

    from app.config import get_settings

    threshold = get_settings().topic_threshold
    out_dir.mkdir(parents=True, exist_ok=True)

    class LogitsOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

    tokenizer = AutoTokenizer.from_pretrained(best, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        best, local_files_only=True, attn_implementation="eager"
    ).eval()
    sample = tokenizer(
        "猫窝买大了想退", return_tensors="pt", max_length=MAX_LENGTH,
        truncation=True, padding="max_length",
    )
    example = (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])
    target = out_dir / "model.onnx"
    started = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            LogitsOnly(model), example, str(target),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"}, "logits": {0: "batch"},
            },
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    export_seconds = time.perf_counter() - started

    # 重新加载一个干净模型来做一致性对比(被 trace 过的实例可能带着副作用)
    del model
    model = AutoModelForSequenceClassification.from_pretrained(
        best, local_files_only=True, attn_implementation="eager"
    ).eval()
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])

    rows = load_rows(DATA / "test.jsonl")
    texts = [row["text"] for row in rows]
    max_abs_diff = 0.0
    mismatch = 0
    for start in range(0, len(texts), 32):
        batch_texts = texts[start:start + 32]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            torch_logits = model(**enc).logits.numpy()
        ort_logits = session.run(["logits"], {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
            "token_type_ids": enc["token_type_ids"].numpy(),
        })[0]
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(torch_logits - ort_logits))))
        torch_probs = 1.0 / (1.0 + np.exp(-torch_logits))
        ort_probs = 1.0 / (1.0 + np.exp(-ort_logits))
        mismatch += int(np.sum((torch_probs >= threshold) != (ort_probs >= threshold)))

    # 推理服务要的 tokenizer 放在模型旁边
    shutil.copy2(best / "tokenizer.json", out_dir / "tokenizer.json")

    report = {
        "model": str(best),
        "onnx": str(target),
        "onnx_bytes": target.stat().st_size,
        "export_seconds": round(export_seconds, 2),
        "checked_rows": len(rows),
        "max_abs_logit_diff": max_abs_diff,
        "threshold": threshold,
        "threshold_mismatches": mismatch,
        "parity_passed": bool(max_abs_diff < 1e-4 and mismatch == 0),
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["parity_passed"]:
        print("ONNX 与 PyTorch 不一致,别拿去上线", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
