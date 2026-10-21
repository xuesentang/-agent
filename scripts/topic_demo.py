"""主题分类器手测（第十章验收 3 用）。

    python scripts/topic_demo.py multi                 # 多诉求句是否命中多类
    python scripts/topic_demo.py predict "买大了想退"    # 单句看分数
    python scripts/topic_demo.py taxonomy              # 17 类 + 边界说明

只加载 ONNX（onnxruntime + tokenizers，不带 torch），和线上旁路用的是同一个
`TopicClassifier`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MULTI_CASES = [
    ("买大了想退", {"退换货", "尺码"}),
    ("发票开错了，顺便问下能不能改收货地址", {"发票", "订单修改"}),
    ("退货运费谁出，优惠券还退吗", {"运费", "优惠活动"}),
    ("这个包能退货吗，保修期内坏了修不修", {"退换货", "保修维修"}),
]


def _fmt(scores: dict, limit: int = 4) -> str:
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
    return " ".join(f"{label}:{score:.2f}" for label, score in top)


def main() -> int:
    parser = argparse.ArgumentParser(description="主题分类器手测")
    parser.add_argument("mode", choices=["multi", "predict", "taxonomy"])
    parser.add_argument("text", nargs="?", default="", help="predict 模式的输入")
    args = parser.parse_args()

    if args.mode == "taxonomy":
        from app.ml.topic.taxonomy import BOUNDARIES, LABELS, NEIGHBOR_RULES

        for label in LABELS:
            print(f"{label:<6}{BOUNDARIES[label]}")
        print(f"\n划界口径：{NEIGHBOR_RULES}")
        return 0

    from app.ml.topic.classifier import load_classifier

    try:
        classifier = load_classifier()
    except Exception as exc:  # noqa: BLE001 - 缺模型/缺依赖都要说清楚
        print(f"分类器不可用：{exc}")
        print("先跑 python scripts/fetch_topic_model.py --from <lab 目录>")
        return 1

    if args.mode == "predict":
        if not args.text.strip():
            print("predict 需要一句原话")
            return 1
        pred = classifier.predict([args.text])[0]
        print(f"{args.text}\n  → {'、'.join(pred.labels) or '(全部低于阈值)'}\n  {_fmt(pred.scores)}")
        return 0

    # multi：逐句判定「是否命中 ≥2 类」，并核对关键标签有没有漏
    texts = [text for text, _ in MULTI_CASES]
    predictions = classifier.predict(texts)
    passed = 0
    for (text, expect), pred in zip(MULTI_CASES, predictions):
        hit = set(pred.labels)
        ok = len(hit) >= 2 and expect <= hit
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        print(f"       标签 {'、'.join(pred.labels) or '(空)'}   分数 {_fmt(pred.scores)}")
        if not ok:
            print(f"       期望至少包含 {'、'.join(sorted(expect))} 且 ≥2 类")
    print(f"\n多诉求 {passed}/{len(MULTI_CASES)} 通过")
    return 0 if passed == len(MULTI_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
