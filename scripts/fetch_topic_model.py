"""把微调好的主题分类模型从 finetune-lab 拉到本项目(第十章)。

模型 409MB,不进 git。这个脚本负责「搬 + 校验」:哈希对不上直接失败退出,
半截文件比没有文件更难查(ONNX 能加载一半、推理出离谱结果,还不报错)。

    python scripts/fetch_topic_model.py                      # 默认从 ../python-kefu-agent/finetune-lab 拉
    python scripts/fetch_topic_model.py --from D:/path/to/lab --force

拉什么:
    <lab>/experiments/onnx/model.onnx            → models/topic_classifier/model.onnx
    <lab>/experiments/best_model/tokenizer.json  → models/topic_classifier/tokenizer.json
    tokenizer_config.json                         → 同上(仅参考,推理不读)
校验:<lab>/SHA256SUMS.txt 里记的哈希(模型与 labels.json)。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAB = Path("D:/python-kefu-agent/finetune-lab")
DEFAULT_OUT = ROOT / "models" / "topic_classifier"

# lab 里算出来、本脚本要逐字节对上的文件(相对 lab 根的路径 → 校验规则)
TRANSFER = {
    "experiments/onnx/model.onnx": "model.onnx",
    "experiments/best_model/tokenizer.json": "tokenizer.json",
    "experiments/best_model/tokenizer_config.json": "tokenizer_config.json",
}

# 想重训才需要基座(411MB):--with-base 时一并搬
BASE_FILES = (
    "config.json",
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sums(lab: Path) -> dict[str, str]:
    path = lab / "SHA256SUMS.txt"
    if not path.exists():
        return {}
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0].strip()
    return sums


def main() -> int:
    parser = argparse.ArgumentParser(description="拉主题分类模型资产")
    parser.add_argument("--from", dest="src", default=str(DEFAULT_LAB), help="finetune-lab 目录")
    parser.add_argument("--to", dest="dst", default=str(DEFAULT_OUT), help="目标目录")
    parser.add_argument("--force", action="store_true", help="已存在也重新拷")
    parser.add_argument(
        "--with-base", action="store_true", help="连基座 RoBERTa 一起搬(重训才需要,411MB)"
    )
    args = parser.parse_args()

    lab = Path(args.src)
    out = Path(args.dst)
    if not lab.exists():
        print(f"找不到实验目录:{lab}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    sums = read_sums(lab)

    for rel, name in TRANSFER.items():
        src = lab / rel
        if not src.exists():
            print(f"缺文件:{src}", file=sys.stderr)
            return 2
        target = out / name
        if target.exists() and not args.force and target.stat().st_size == src.stat().st_size:
            print(f"已存在,跳过:{name}({target.stat().st_size} 字节;--force 强制重拷)")
            continue
        shutil.copy2(src, target)
        print(f"拷贝 {name}:{target.stat().st_size} 字节")

    # 校验:只校 SHA256SUMS.txt 里有的条目
    checked = 0
    for rel, name in TRANSFER.items():
        expect = sums.get(rel)
        if not expect:
            continue
        digest = sha256_of(out / name)
        checked += 1
        if digest != expect:
            print(f"哈希不符:{name}\n  期望 {expect}\n  实际 {digest}", file=sys.stderr)
            return 1
        print(f"校验通过:{name}")

    print(f"\n模型就绪:{out}")
    if not checked:
        print("(lab 没提供 SHA256SUMS.txt,跳过校验——这次没有可对照的哈希)")

    if args.with_base:
        base_src = lab / "models" / "chinese-roberta-wwm-ext"
        base_dst = ROOT / "models" / "chinese-roberta-wwm-ext"
        if not base_src.exists():
            print(f"找不到基座:{base_src}", file=sys.stderr)
            return 2
        base_dst.mkdir(parents=True, exist_ok=True)
        for name in BASE_FILES:
            src = base_src / name
            if src.exists():
                shutil.copy2(src, base_dst / name)
        print(f"基座就绪:{base_dst}(重训用:python scripts/train_topic_model.py full)")
    else:
        print("只拉了推理资产。要重训再加 --with-base(基座 411MB)。")

    print("下一步:TOPIC_ENABLED=true python scripts/classify_topics.py --batch 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
