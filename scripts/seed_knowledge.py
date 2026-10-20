"""生成 RAG 种子知识文档(AI 合成数据)—— 三步建库流水线的第一步。

用法:
    python scripts/seed_knowledge.py                  # 写入 data/knowledge_docs/(已存在则跳过)
    python scripts/seed_knowledge.py --force          # 覆盖重写
    python scripts/seed_knowledge.py --target DIR     # 指定目录

第二步建库:python scripts/build_knowledge.py --docs data/knowledge_docs
文档内容内嵌于 app/rag/seed_docs.py(单一事实源),本脚本只负责落盘。
"""

import argparse
import sys
from pathlib import Path

# 从仓库任意位置跑(python scripts/seed_knowledge.py)时,把项目根挂上 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.seed_docs import write_seed_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAG 种子知识文档(AI 合成)")
    parser.add_argument("--target", default="data/knowledge_docs", help="输出目录")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文档")
    args = parser.parse_args()

    written, skipped = write_seed_docs(Path(args.target), force=args.force)
    for path in written:
        print(f"written {path}")
    for path in skipped:
        print(f"skipped {path}(已存在,--force 覆盖)")
    print(f"done: {len(written)} written, {len(skipped)} skipped -> {args.target}")


if __name__ == "__main__":
    main()
