"""离线建库入口:文档目录 → 切分 → MySQL pending → BGE-M3 嵌入 → Milvus upsert → 回填 done。

用法:
    python scripts/build_knowledge.py                          # 默认 data/knowledge_docs
    python scripts/build_knowledge.py --docs DIR --batch-size 16
    python scripts/build_knowledge.py --stop-after 1           # 中断演练:跑一批即停,重跑补齐
    python scripts/build_knowledge.py --fake-embedder          # 冒烟:不加载真模型(确定性假向量)
    python scripts/build_knowledge.py --pending-only           # 只补跑 pending(不读文档)

前置:MySQL 已 apply scripts/sql/ch02+ch03 schema;向量库走 VECTOR_STORE_URI
(默认 ./data/milvus_lite.db,Milvus Lite 文件库,切服务端只改 .env 一行)。

中断演练(验收标准 2):--stop-after 1 跑一批后抛 BuildInterrupted 退出;
不带 --stop-after 重跑,只捞 pending 块补向量化,Milvus 按主键 upsert 不重复。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 从仓库任意位置跑(python scripts/build_knowledge.py)时,把项目根挂上 syspath。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.rag.doc_loader import load_docs
from app.rag.embeddings import FakeEmbedder, LocalBGEEmbedder
from app.rag.indexer import BuildInterrupted, build_index
from app.rag.vectorstore import MilvusStore


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 离线建库(MySQL 权威源 + Milvus 向量)")
    parser.add_argument("--docs", default="data/knowledge_docs", help="知识文档目录或单个 .md 文件")
    parser.add_argument("--batch-size", type=int, default=32, help="每批向量化的块数")
    parser.add_argument("--stop-after", type=int, default=None, help="中断演练:跑完 N 批后停止")
    parser.add_argument("--fake-embedder", action="store_true", help="用确定性假向量冒烟(不加载真模型)")
    parser.add_argument("--pending-only", action="store_true", help="不读文档,只补跑 pending 块")
    parser.add_argument("--reset-store", action="store_true", help="先删 Milvus knowledge 集合再重建(清存量向量)")
    parser.add_argument("--database-url", default=get_settings().database_url, help="覆盖 DATABASE_URL")
    parser.add_argument("--vector-store-uri", default=None, help="覆盖 VECTOR_STORE_URI(如另起一个 Lite 文件库)")
    args = parser.parse_args()

    docs = [] if args.pending_only else load_docs(Path(args.docs))
    if not args.pending_only and not docs:
        print(f"没有找到可建库的文档:{args.docs}(先跑 scripts/seed_knowledge.py 生成种子文档)")
        sys.exit(1)

    embedder = FakeEmbedder() if args.fake_embedder else LocalBGEEmbedder()
    store = MilvusStore(uri=args.vector_store_uri)
    if args.reset_store:
        store.drop_collection()
        print("reset: dropped collection knowledge(重建后为 FLAT 索引)")
    engine = create_async_engine(args.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    print(f"docs={len(docs)} batch_size={args.batch_size} "
          f"embedder={'fake' if args.fake_embedder else 'LocalBGE-M3'} "
          f"db={args.database_url} store={args.vector_store_uri or get_settings().vector_store_uri}")

    async def run():
        try:
            if args.reset_store:
                # 集合已经 drop,MySQL 里的 vector_id 必须一起作废,否则这些块
                # 会被当成「已向量化」永远跳过(Milvus 少一截,而且不报错)。
                from app.repositories import knowledge_repo

                async with session_factory() as session:
                    n = await knowledge_repo.reset_vector_status(session)
                print(f"reset: {n} 块退回 pending(与已删的向量集合对齐)")
            return await build_index(
                session_factory=session_factory,
                store=store,
                embedder=embedder,
                docs=docs,
                batch_size=args.batch_size,
                stop_after_batches=args.stop_after,
            )
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(run())
    except BuildInterrupted as exc:
        print(f"interrupted: {exc}")
        print("重跑(不带 --stop-after)自动补齐剩余 pending")
        return
    print(f"done: {report.summary()}")


if __name__ == "__main__":
    main()
