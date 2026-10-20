"""对话挖知识入口:历史 messages → LLM 抽 QA → staging 暂存 → 去重 → knowledge_chunks。

用法:
    python scripts/mine_qa.py                          # 默认分批 20 会话
    python scripts/mine_qa.py --batch-size 10 --max-batches 1
    python scripts/mine_qa.py --dry-run                # 只预览能抽出什么,不写库
    python scripts/mine_qa.py --database-url ...       # 覆盖 DATABASE_URL

抽完跑一次 `python scripts/build_knowledge.py --pending-only` 补向量化,
挖出来的 QA 才进 Milvus 可被检索。定时执行走服务端 MINE_QA_ENABLED 开关
(默认关),见 app/main.py lifespan。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 从仓库任意位置跑(python scripts/mine_qa.py)时,把项目根挂上 syspath。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.rag.miner import mine_qa_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="从历史对话挖 QA 知识对(GLM 抽取)")
    parser.add_argument("--batch-size", type=int, default=get_settings().mine_qa_batch_size)
    parser.add_argument("--max-batches", type=int, default=None, help="最多跑几批(默认全部)")
    parser.add_argument("--dry-run", action="store_true", help="只抽取和统计,不写任何表")
    parser.add_argument("--database-url", default=get_settings().database_url)
    args = parser.parse_args()

    engine = create_async_engine(args.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def run():
        try:
            return await mine_qa_pairs(
                session_factory=session_factory,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                dry_run=args.dry_run,
            )
        finally:
            await engine.dispose()

    report = asyncio.run(run())
    print(f"done: {report.summary()}")
    if report.kept:
        print("kept 的 QA 已入 knowledge_chunks(pending);跑 "
              "`python scripts/build_knowledge.py --pending-only` 补向量化")


if __name__ == "__main__":
    main()
