"""Apply DDL scripts to MySQL(scripts/sql/*.sql)。

用法(需先 docker compose up -d 起好 MySQL):
    python scripts/apply_schema.py                    # 默认按文件名序执行全部 schema(ch02 → ch03 → …)
    python scripts/apply_schema.py --sql <path>       # 只执行指定 SQL 文件

注意:按分号切分执行,适用于本仓库纯 DDL 脚本(无存储过程/触发器);
后续章节若引入含分号字符串或例程的脚本,需换成逐语句解析。
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

# 从仓库任意位置跑(python scripts/apply_schema.py)时,把项目根挂上 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

SCHEMA_DIR = Path(__file__).resolve().parent / "sql"


def default_schema_files() -> list[Path]:
    """按文件名序(章节序)返回全部 schema 脚本;增量章节追加新文件即可。"""
    return sorted(SCHEMA_DIR.glob("*.sql"))


def split_statements(sql: str) -> list[str]:
    """按分号切分,但先剥离 `--` 行注释——DDL 注释里可能含半角分号
    (ch02 的 faq 表注释就有),直接 split 会把注释切碎导致语法错误。

    适用边界:本仓库纯 DDL 脚本,无块注释、无字符串字面量含 `--`;
    后续章节若引入例程/触发器,需换真正的语句解析器。
    """
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


async def apply(database_url: str, sql_path: str, if_not_exists: bool = False) -> int:
    sql = Path(sql_path).read_text(encoding="utf-8")
    statements = split_statements(sql)
    if if_not_exists:
        # 幂等重跑:CREATE TABLE → CREATE TABLE IF NOT EXISTS(不动表内数据)。
        statements = [
            re.sub(r"(?i)CREATE TABLE(?! IF NOT EXISTS)", "CREATE TABLE IF NOT EXISTS", s)
            for s in statements
        ]
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            for stmt in statements:
                await conn.execute(text(stmt))
        return len(statements)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply DDL to MySQL")
    parser.add_argument("--database-url", default=get_settings().database_url)
    parser.add_argument(
        "--sql",
        default=None,
        help="只执行指定 SQL 文件;默认按文件名序执行 scripts/sql/*.sql 全部",
    )
    parser.add_argument(
        "--if-not-exists",
        action="store_true",
        help="幂等重跑:CREATE TABLE 转为 CREATE TABLE IF NOT EXISTS",
    )
    args = parser.parse_args()
    files = [Path(args.sql)] if args.sql else default_schema_files()
    total = 0
    for f in files:
        total += asyncio.run(apply(args.database_url, str(f), args.if_not_exists))
        print(f"applied {f}")
    print(f"total {total} statements across {len(files)} file(s)")


if __name__ == "__main__":
    main()
