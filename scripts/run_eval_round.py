"""跑一轮评估并落 `eval_runs`（ch09 需求 6）。

    python scripts/run_eval_round.py                      # 只跑检索段（默认 hybrid_rerank）
    python scripts/run_eval_round.py --strategy hybrid     # 换策略看趋势
    python scripts/run_eval_round.py --limit 10            # 只跑前十题（冒烟）
    python scripts/run_eval_round.py --no-rerank           # 不加载重排模型（省内存）

跑完去看趋势：

    curl localhost:8000/api/v1/review/trend
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
    from app.config import get_settings
    from app.db import get_sessionmaker
    from app.services import eval_service

    retriever = None
    if not args.no_rerank:
        from app.rag.embeddings import LocalBGEEmbedder
        from app.rag.reranker import LocalBGEReranker
        from app.rag.retriever import Retriever
        from app.rag.vectorstore import MilvusStore

        print("加载嵌入与重排模型……")
        retriever = Retriever(
            store=MilvusStore(), embedder=LocalBGEEmbedder(), reranker=LocalBGEReranker()
        )
    else:
        print("不加载重排模型：策略 hybrid_rerank 会退化，只能看个大概")

    result = await eval_service.run_round(
        get_sessionmaker(),
        cases_path=args.cases or get_settings().eval_cases_path,
        retriever=retriever,
        strategy=args.strategy,
        triggered_by=args.triggered_by,
        top_k=args.top_k,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        m = result["metrics"]
        print(
            f"第 {result['id']} 轮：{m['cases']} 题 | R@5={m['r5']:.3f} R@10={m['r10']:.3f} "
            f"MRR={m['mrr']:.3f}（干净 MRR={m['clean_mrr']:.3f}）| 加权 MRR={m['wmrr']:.3f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="跑一轮检索评估并写入 eval_runs")
    parser.add_argument("--cases", default=None, help="用例文件（默认取配置 EVAL_CASES_PATH）")
    parser.add_argument("--strategy", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--triggered-by", default="手动", choices=["手动", "定时"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
