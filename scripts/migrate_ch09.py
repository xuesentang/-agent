"""把 ch09 的表与列补到当前库（开发台 SQLite 与 MySQL 通用）。

    python scripts/migrate_ch09.py            # 建新表 + 给池表加两列，已存在就跳过
    python scripts/migrate_ch09.py --dry-run  # 只打印要做什么

为什么不直接跑 `scripts/sql/ch09_schema.sql`：那份是 MySQL 语法（ENUM / JSON / ENGINE），
开发台是 SQLite。这里走 SQLAlchemy 的反射——新表交给 `Base.metadata.create_all`，
老表缺的列用 `ALTER TABLE ... ADD COLUMN` 补，两边都能跑，且重复执行安全。

生产/真 MySQL 环境仍建议跑那份 .sql（字段注释、字符集、外键行为都在那儿）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 池表在 ch04 就建好了，新加的两列要按 ALTER 走；类型按方言粗分即可
NEW_COLUMNS = {
    "retrieved_chunks": {"sqlite": "JSON", "mysql": "JSON", "default": "JSON"},
    "matched_review_id": {"sqlite": "BIGINT", "mysql": "BIGINT UNSIGNED", "default": "BIGINT"},
}


async def _existing_columns(conn, table: str) -> set[str]:
    from sqlalchemy import inspect

    return await conn.run_sync(lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns(table)})


async def main_async(dry_run: bool) -> int:
    from sqlalchemy import text

    from app.db import get_engine
    from app.models import Base

    engine = get_engine()
    dialect = engine.dialect.name

    async with engine.begin() as conn:
        existing_tables = await conn.run_sync(lambda c: set(c.dialect.get_table_names(c)))

        # 1) 新表：create_all 只建缺的，不动已有的
        if dry_run:
            new_tables = sorted(set(Base.metadata.tables) - existing_tables)
            print(f"[dry-run] 将新建表：{new_tables or '无'}")
        else:
            await conn.run_sync(Base.metadata.create_all)
            print(f"[{dialect}] 新表已就绪（create_all 跳过已存在的）")

        # 2) 老表加列
        columns = await _existing_columns(conn, "low_confidence_questions")
        for name, types in NEW_COLUMNS.items():
            if name in columns:
                print(f"列已存在，跳过：low_confidence_questions.{name}")
                continue
            col_type = types.get(dialect, types["default"])
            stmt = f"ALTER TABLE low_confidence_questions ADD COLUMN {name} {col_type} NULL"
            if dry_run:
                print(f"[dry-run] {stmt}")
                continue
            await conn.execute(text(stmt))
            print(f"已加列：{name} {col_type}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="补 ch09 表与列")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
