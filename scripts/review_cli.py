"""审核队列的命令行入口（无浏览器时的可用路径）。

    python scripts/review_cli.py list                 # 看待审（按出现次数排）
    python scripts/review_cli.py show 12              # 看详情：原话 + 召回快照
    python scripts/review_cli.py approve 12 --answer "签收后 7 天内可申请退货，运费……"
    python scripts/review_cli.py reject 12
    python scripts/review_cli.py approve 12 --answer-file answer.txt

审核通过走的是 ch03 的双写流程（建 pending 块 → 嵌入 → upsert 向量库 → 回填 done），
所以这一条命令要加载嵌入模型，跑起来比 list/show 慢。

数据源与向量库默认指开发台的 SQLite + Milvus Lite（与 tool_demo.py 一致）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")
os.environ.setdefault("VECTOR_STORE_URI", "./data/ch07_milvus.db")


async def _show_detail(session, review_id: int) -> int:
    from app.repositories import low_confidence_repo, review_repo

    row = await review_repo.get(session, review_id)
    if row is None:
        print(f"没有 #{review_id} 这条缺口")
        return 1
    print(f"#{row.id} [{row.review_status}] 出现 {row.occurrence_count} 次")
    print(f"问题：{row.normalized_question}")
    print(f"AI 示例答案：{row.ai_suggested_answer or '（无）'}")
    if row.approved_answer:
        print(f"已核准答案：{row.approved_answer}")

    sources = await low_confidence_repo.list_by_review(session, review_id)
    print(f"\n贡献这条缺口原话 {len(sources)} 条：")
    for q in sources:
        print(f"  - #{q.id} [{q.source}] {q.raw_question}")
        for chunk in q.retrieved_chunks or []:
            print(f"      快照 {chunk.get('score')} {chunk.get('section_path')} | {chunk.get('snippet')}")
    return 0


async def main_async(args) -> int:
    from app.db import get_sessionmaker
    from app.services import flywheel

    maker = get_sessionmaker()

    if args.action == "list":
        from app.repositories import review_repo

        async with maker() as session:
            rows = await review_repo.list_for_dedupe(session, limit=args.limit)
            counts = await review_repo.counts(session)
        print(f"待审 {len(rows)} 条 / 全库 {counts}")
        for r in rows:
            print(f"  #{r.id:>4} ×{r.occurrence_count:<3} {r.normalized_question[:60]}")
            if r.ai_suggested_answer and args.verbose:
                print(f"        示例答案：{r.ai_suggested_answer[:80]}")
        return 0

    if args.action == "show":
        async with maker() as session:
            return await _show_detail(session, args.review_id)

    if args.action == "reject":
        async with maker() as session:
            row = await flywheel.reject(session, args.review_id)
        if row is None:
            print(f"没有 #{args.review_id} 这条缺口")
            return 1
        print(f"#{row.id} 已驳回")
        return 0

    # approve：要加载嵌入模型，所以放在最后
    answer = args.answer
    if args.answer_file:
        answer = Path(args.answer_file).read_text(encoding="utf-8").strip()
    if not answer:
        print("通过审核必须给答案：--answer 或 --answer-file")
        return 2

    from app.rag.embeddings import LocalBGEEmbedder
    from app.rag.vectorstore import MilvusStore

    embedder = LocalBGEEmbedder()
    store = MilvusStore()
    async with maker() as session:
        try:
            result = await flywheel.approve(
                session, args.review_id, answer, store=store, embedder=embedder
            )
        except ValueError as exc:
            print(f"没法通过：{exc}")
            return 2
    if result is None:
        print(f"没有 #{args.review_id} 这条缺口")
        return 1
    print(
        f"#{result['review_id']} 已通过 → 知识块 #{result['chunk_id']}"
        f"（{'新建' if result['created'] else '改答案'}，本轮补向量化 {result['vectorized']} 条）"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="飞轮待审队列：查看 / 通过 / 驳回")
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="看待审队列")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--verbose", action="store_true", help="带上示例答案摘要")

    for name, help_text in (("show", "看详情"), ("approve", "通过并写知识库"), ("reject", "驳回")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("review_id", type=int)

    p_approve = sub.choices["approve"]
    p_approve.add_argument("--answer", default="", help="核准答案")
    p_approve.add_argument("--answer-file", default="", help="从文件读核准答案")

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
