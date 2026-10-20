"""Run the ch03 paraphrase-recall eval against the real stack (BGE-M3 + Milvus).

query → 向量检索 Top-5 → 命中判定。判定规则(与 query_faq 的 top_k=5 一致):
Top-5 里存在任一 chunk 同时满足
  ① expect_section 是其 section_path 的子串,或该 chunk 是对话挖出来的 QA
     (content_type=qa_mined——挖出的 QA 正面回答同一问题,算命中);
  ② expect_keyword 在其 answer 里。
Usage:
    python -m tests.eval.run_recall_eval [--database-url ...] [--vector-store-uri ...]
Defaults to the project stack (settings.database_url + VECTOR_STORE_URI).
Exit code 0 iff pass rate >= 90%.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

CASES_PATH = Path(__file__).parent / "ch03_recall_cases.jsonl"
PASS_THRESHOLD = 0.9
TOP_K = 5


def score_case(case: dict, hits: list) -> bool:
    for hit in hits:
        chunk = hit.chunk
        if case["expect_keyword"] not in chunk.answer:
            continue
        section = chunk.section_path or ""
        if case["expect_section"] in section or chunk.content_type == "qa_mined":
            return True
    return False


async def run(database_url: str, vector_store_uri: str | None) -> int:
    from app.config import get_settings
    from app.rag.embeddings import LocalBGEEmbedder
    from app.rag.retriever import Retriever, retrieve
    from app.rag.vectorstore import MilvusStore
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    retriever = Retriever(store=MilvusStore(uri=vector_store_uri), embedder=LocalBGEEmbedder())

    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    try:
        for i, case in enumerate(cases, 1):
            try:
                hits = await retrieve(
                    session_factory=session_factory,
                    query=case["query"],
                    top_k=TOP_K,
                    retriever=retriever,
                )
            except Exception as exc:  # noqa: BLE001 - eval reports, never crashes
                hits = []
                print(f"[FAIL] #{i} {case['query']} error:{exc}")
                continue
            ok = score_case(case, hits)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            top = hits[0].chunk if hits else None
            top_desc = (
                f"{(top.section_path or top.questions)[:30]}..." if top else "(空)"
            )
            print(f"[{status}] #{i} {case['query']} -> top1:{top_desc} 期望:{case['expect_section']}")
    finally:
        await engine.dispose()
    rate = passed / len(cases) if cases else 0.0
    print(f"\n{passed}/{len(cases)} passed, rate={rate:.0%}, threshold={PASS_THRESHOLD:.0%}")
    return 0 if rate >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="ch03 换说法召回评估(真 BGE-M3 + Milvus)")
    parser.add_argument("--database-url", default=get_settings().database_url)
    parser.add_argument("--vector-store-uri", default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.database_url, args.vector_store_uri)))
