"""人工抽检素材生成(P1-2)。

为什么要单独跑一遍:
    评估报告里的答案被截到 60 字,拿截断文本没法判「有没有编」。抽检必须看到**完整答案**
    和**它引用的证据原文**,否则就是拿裁判的话当真相——而这一轮裁判和生成是同一个模型,
    它给自己打分本来就可疑。

这个脚本只打印材料,不给结论。结论由人看完写进 dev-notes。

用法:
    python scripts/spotcheck_eval.py --ids A06,B11,E27,C20,D12
    python scripts/spotcheck_eval.py --auto 12          # 按桶轮转挑 12 条
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.llm import get_chat_model, get_judge_model
from app.rag.citations import build_citations, format_evidence
from app.rag.embeddings import LocalBGEEmbedder
from app.rag.reranker import LocalBGEReranker
from app.rag.retriever import Retriever, retrieve
from app.rag.vectorstore import MilvusStore
from tests.eval.run_rag_eval import GENERATE_PROMPT, judge_answer

CASES = Path("tests/eval/ch04_cases.jsonl")
OUT = Path("reports/ch04_spotcheck.md")
K = 10


def pick(cases: list[dict], auto: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for c in cases:
        if c.get("noise"):
            continue
        buckets.setdefault(c["bucket"], []).append(c)
    picked: list[dict] = []
    i = 0
    while len(picked) < auto:
        progressed = False
        for bucket in sorted(buckets):
            if i < len(buckets[bucket]) and len(picked) < auto:
                picked.append(buckets[bucket][i])
                progressed = True
        i += 1
        if not progressed:
            break
    return picked


async def main_async(args) -> int:
    cases = [
        json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        picked = [c for c in cases if c["id"] in want]
    else:
        picked = pick([c for c in cases if c.get("answerable")], args.auto)

    settings = get_settings()
    engine = create_async_engine(args.database_url or settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    retriever = Retriever(
        store=MilvusStore(uri=args.vector_store_uri),
        embedder=LocalBGEEmbedder(),
        reranker=LocalBGEReranker(),
    )
    model = get_chat_model()
    judge_model, _ = get_judge_model()

    lines = ["# ch04 人工抽检材料", "", f"抽检 {len(picked)} 条,每条给完整答案与引用证据。", ""]
    for case in picked:
        hits = await retrieve(
            session_factory=session_factory,
            query=case["query"],
            top_k=K,
            strategy="hybrid_rerank",
            retriever=retriever,
            rewrite=False,
        )
        citations = build_citations(hits)
        evidence = format_evidence(citations)
        points = case.get("expected_keywords") or []
        if case.get("answerable"):
            answer = (
                await model.ainvoke(
                    [
                        (
                            "human",
                            GENERATE_PROMPT.format(
                                system="你是电商客服,依据证据回答。",
                                evidence=evidence,
                                question=case["query"],
                            ),
                        )
                    ]
                )
            ).content
            verdict = await judge_answer(judge_model, evidence, case["query"], points, answer)
            verdict_text = (
                f"裁判:faithful={verdict['faithful']} 覆盖 "
                f"{verdict['covered_count']}/{verdict['relevant_count']}(本节共 {len(points)} 条)"
            )
        else:
            answer = "(不可答题,不生成)"
            verdict_text = ""

        lines += [
            f"## {case['id']} {case['bucket']} {case['query']}",
            "",
            f"- 标注:{case['expected_section'] or '—(不可答)'}",
            f"- 要点:{';'.join(points) or '—'}",
            f"- {verdict_text}" if verdict_text else "- (不可答)",
            "",
            "**答案**",
            "",
            answer.strip(),
            "",
            "**证据(前 4 条截断)**",
            "",
        ]
        for c in citations[:4]:
            lines += [f"- [{c.n}] {c.section_path} · {c.question}", "```", c.answer[:400].strip(), "```"]
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}:{len(picked)} 条")
    await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="人工抽检材料")
    parser.add_argument("--ids", default="")
    parser.add_argument("--auto", type=int, default=12)
    parser.add_argument("--database-url", default="sqlite+aiosqlite:///./data/devstack.db")
    parser.add_argument("--vector-store-uri", default="./data/devstack_milvus.db")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
