"""主题归类批量任务（第十章需求 5）。

    # 先看模型怎么标（不写库）
    python scripts/classify_topics.py --dry-run --batch 20

    # 真跑一批
    python scripts/classify_topics.py --batch 50

    # 定时任务口径：池里攒够 50 条才跑，不够就什么都不做（退出码 0）
    python scripts/classify_topics.py --min-batch 50

退出码：0 正常（含"没事可做"）；1 分类器不可用（缺模型/缺依赖）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 开发台默认：和 CH07/CH08 同一套库（池子里的问题在那儿）
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")


def _print_distribution(dist: dict, total: int) -> None:
    if not dist:
        print("(没有可统计的类目)")
        return
    print(f"\n{'类目':<8}{'数量':>6}  占比")
    for label, count in dist.items():
        share = f"{count / total * 100:5.1f}%" if total else "-"
        bar = "█" * min(40, int(count / max(1, max(dist.values())) * 40))
        print(f"{label:<8}{count:>6}  {share} {bar}")


async def main_async(args) -> int:
    from app.db import get_engine, get_sessionmaker
    from app.models import Base
    from app.services.topic_service import classify_pending

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # 表不在就先建,脚本不依赖迁移

    result = await classify_pending(
        get_sessionmaker(),
        batch_size=args.batch,
        min_batch=args.min_batch,
        dry_run=args.dry_run,
    )

    if args.dry_run and result.get("preview"):
        print(json.dumps(result["preview"], ensure_ascii=False, indent=2))
        print(f"\n共 {result['picked']} 条(未写库)")
        return 0

    if result["reason"]:
        print(f"跳过:{result['reason']}")
        return 1 if "分类器不可用" in result["reason"] else 0

    print(
        f"取到 {result['picked']} 条,写入 {result['classified']} 条"
        + ("(同一问题重复归类只更新标签)" if result["classified"] else "")
    )
    if result["distribution"]:
        _print_distribution(result["distribution"], result["classified"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="低置信度问题批量主题归类")
    parser.add_argument("--batch", type=int, default=None, help="这一批最多取多少条")
    parser.add_argument(
        "--min-batch", type=int, default=1, help="攒够多少条才跑(定时任务用,默认 1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="只算不写库")
    parser.add_argument("--verbose", action="store_true", help="打 INFO 日志")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
