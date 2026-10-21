"""跑一轮飞轮：池里未归并的原话 → 标准化 → 查重 → 进待审队列。

    python scripts/run_flywheel.py --batch 50
    python scripts/run_flywheel.py --batch 50 --dry-run   # 只算不写，看会新增几条缺口
    python scripts/run_flywheel.py --since-id 120         # 续跑：从这一行之后接着处理

数据源与向量库默认指向开发台的 SQLite + Milvus Lite（与 tool_demo.py 一致），
换真库改 .env 里的 DATABASE_URL / VECTOR_STORE_URI 即可。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")
os.environ.setdefault("VECTOR_STORE_URI", "./data/ch07_milvus.db")


async def main_async(args) -> int:
    from app.core.llm import get_extract_model
    from app.db import get_sessionmaker
    from app.services import flywheel

    model = get_extract_model()
    stats = await flywheel.run_batch(
        get_sessionmaker(),
        limit=args.batch,
        model=model,
        dry_run=args.dry_run,
        since_id=args.since_id,
    )

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        head = "预览（未落库）" if stats["dry_run"] else "本轮"
        print(
            f"{head}：处理 {stats['processed']} 条 / 新建缺口 {stats['created']} / "
            f"累加 {stats['merged']} / 失败 {stats['failed']} / 最后 id {stats['last_id']}"
        )
        for it in stats["items"]:
            extra = it.get("detail") or (f"→ #{it['review_id']}" if it.get("review_id") else "")
            print(f"  [{it['via']:>6}] #{it['id']} {it['question']} {extra}")

    if args.batch and stats["processed"] >= args.batch and not args.dry_run:
        print(f"\n还有剩余，续跑：--since-id {stats['last_id']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="跑一轮数据飞轮（标准化 + 查重 + 入待审队列）")
    parser.add_argument("--batch", type=int, default=50, help="这一批处理多少条（默认 50）")
    parser.add_argument("--since-id", type=int, default=0, help="从哪个池内 id 之后开始（续跑用）")
    parser.add_argument("--dry-run", action="store_true", help="只算不写")
    parser.add_argument("--json", action="store_true", help="输出 JSON（给后续流水线用）")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
