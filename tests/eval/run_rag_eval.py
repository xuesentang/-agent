"""ch04 检索评估:四策略 × 分桶 Recall@K / MRR + 生成段 Faithfulness。

用法(需要真 BGE-M3 + 真 Milvus + 真模型;先建库):
    python -m tests.eval.run_rag_eval                       # 全跑(检索 + 生成)
    python -m tests.eval.run_rag_eval --no-generation       # 只算检索指标(快)
    python -m tests.eval.run_rag_eval --rewrite             # 打开问法改写再跑一遍
    python -m tests.eval.run_rag_eval --out reports/ch04.md

判定口径:
- 检索:命中 = 期望章节名是命中块 section_path 的子串(分桶统计 Recall@5 / R@10 / MRR);
- 生成:Faithfulness 由 LLM 当裁判,判「答案里的每条事实是否都能在给定证据里找到」,
  判不忠实的个案写进 faith_cases 台账(一题一行,复发计数);
- 拒答:对 answerable=false 的题,走生产同一条闸(检索 → 证据自评),看是否拒答。

关掉改写是刻意的:策略对比要在同一套问法上比,否则分不清分数差异来自策略还是来自改写;
改写的收益单独用 --rewrite 跑一份对照。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.prompts import CUSTOMER_SERVICE_TOOLS_SYSTEM_PROMPT
from app.rag.citations import build_citations, format_evidence
from app.rag.embeddings import LocalBGEEmbedder
from app.rag.reranker import LocalBGEReranker
from app.rag.retriever import Retriever, retrieve
from app.rag.selfcheck import check_evidence
from app.rag.vectorstore import MilvusStore

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
K5, K10 = 5, 10


@dataclass
class CaseResult:
    case_id: str
    bucket: str
    rank: int | None  # 第一条命中块的排名(1-based),None = 前 K 条都没命中
    hits: list = field(default_factory=list)


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(hits, expected_section: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        path = getattr(hit.chunk, "section_path", "") or ""
        if expected_section and expected_section in path:
            return i
    return None


def _recall(results: list[CaseResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.rank is not None and r.rank <= k) / len(results)


def _mrr(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(1.0 / r.rank for r in results if r.rank is not None) / len(results)


# ---------------------------------------------------------------- 生成段


GENERATE_PROMPT = """{system}

下面是检索到的证据:
{evidence}

用户问题:{question}

请依据证据回答,每条结论后标注来源编号(如 [1]);证据不足就直说没查到,不要补充证据之外的内容。"""

JUDGE_PROMPT = """你是忠实度裁判。判断「答案」里的每一条事实性陈述,是否都能在「证据」中找到依据。

规则:
- 只检查事实性内容(规则、数字、流程、时效、费用);语气词、安抚话术不算编造;
- 答案里出现证据没有的规则、数字、承诺,即为不忠实;
- 答案明确说「没查到」「不确定」不算编造。

只输出 JSON:{{"faithful": true 或 false, "reason": "不忠实时指出编在哪一句;忠实则简述依据"}}

证据:
{evidence}

答案:
{answer}"""


async def generate_answer(model, question: str, evidence: str) -> str:
    resp = await model.ainvoke(
        [
            (
                "human",
                GENERATE_PROMPT.format(
                    system=CUSTOMER_SERVICE_TOOLS_SYSTEM_PROMPT,
                    evidence=evidence,
                    question=question,
                ),
            )
        ]
    )
    return (getattr(resp, "content", "") or "").strip()


async def judge_faithful(model, evidence: str, answer: str) -> tuple[bool, str]:
    from app.rag.query_rewrite import _parse_json

    try:
        resp = await model.ainvoke(
            [("human", JUDGE_PROMPT.format(evidence=evidence, answer=answer))],
            response_format={"type": "json_object"},
        )
        payload = _parse_json(getattr(resp, "content", "") or "")
    except Exception as exc:  # noqa: BLE001 - 判不了就当忠实,不算个案
        return True, f"裁判失败:{exc}"
    if "faithful" not in payload:
        return True, "裁判输出不可解析"
    return bool(payload["faithful"]), str(payload.get("reason") or "")


# ---------------------------------------------------------------- 主流程


async def run(args) -> int:
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    answerable = [c for c in cases if c.get("answerable")]
    unanswerable = [c for c in cases if not c.get("answerable")]

    settings = get_settings()
    engine = create_async_engine(args.database_url or settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = MilvusStore(uri=args.vector_store_uri)
    embedder = LocalBGEEmbedder()
    reranker = None if args.no_rerank else LocalBGEReranker(args.rerank_model)
    retriever = Retriever(store=store, embedder=embedder, reranker=reranker)

    print(f"cases={len(cases)} answerable={len(answerable)} buckets={sorted({c['bucket'] for c in cases})}")
    print(f"rewrite={'on' if args.rewrite else 'off'} rerank={'off' if args.no_rerank else 'on'}")

    # 1. 四策略检索指标(--generation-only 跳过,省一次重排全跑)
    per_strategy: dict[str, dict[str, list[CaseResult]]] = {}
    for strategy in (() if args.generation_only else STRATEGIES):
        by_bucket: dict[str, list[CaseResult]] = {}
        for case in answerable:
            hits = await retrieve(
                session_factory=session_factory,
                query=case["query"],
                top_k=K10,
                strategy=strategy,
                retriever=retriever,
                rewrite=args.rewrite,
            )
            rank = _rank(hits, case.get("expected_section") or "")
            by_bucket.setdefault(case["bucket"], []).append(
                CaseResult(case["id"], case["bucket"], rank, hits)
            )
        per_strategy[strategy] = by_bucket
        all_results = [r for rs in by_bucket.values() for r in rs]
        print(
            f"[{strategy:14s}] R@5={_recall(all_results, K5):.3f} "
            f"R@10={_recall(all_results, K10):.3f} MRR={_mrr(all_results):.3f}"
        )

    # 2. 生成段:默认策略 hybrid_rerank
    gen_lines: list[str] = []
    faith_pass = 0
    judged = 0
    unfaithful_cases: list[dict] = []
    if not args.no_generation:
        from app.core.llm import get_chat_model, get_extract_model

        gen_model = get_chat_model()
        judge_model = get_extract_model()
        for case in answerable:
            hits = await retrieve(
                session_factory=session_factory,
                query=case["query"],
                top_k=K10,
                strategy="hybrid_rerank",
                retriever=retriever,
                rewrite=args.rewrite,
            )
            if not hits:
                gen_lines.append(f"- {case['id']} 证据为空,跳过")
                continue
            citations = build_citations(hits)
            evidence = format_evidence(citations)
            answer = await generate_answer(gen_model, case["query"], evidence)
            faithful, reason = await judge_faithful(judge_model, evidence, answer)
            judged += 1
            faith_pass += 1 if faithful else 0
            mark = "OK " if faithful else "!! "
            gen_lines.append(f"- {mark}{case['id']} {case['query']} → {answer[:60]}")
            if not faithful:
                unfaithful_cases.append(
                    {
                        "case": case,
                        "answer": answer,
                        "reason": reason,
                        "citations": [c.to_payload() for c in citations],
                    }
                )
        print(f"[generation] faithfulness={faith_pass}/{judged}")

        if unfaithful_cases and not args.no_ledger:
            from app.repositories import faith_case_repo

            async with session_factory() as session:
                for item in unfaithful_cases:
                    case = item["case"]
                    await faith_case_repo.upsert_case(
                        session,
                        eval_id=case["id"],
                        bucket=case["bucket"],
                        query=case["query"],
                        answer=item["answer"],
                        reason=item["reason"] or "裁判判定不忠实",
                        strategy="hybrid_rerank",
                        citations=item["citations"],
                        judge_model=settings.llm_model,
                    )
            print(f"[ledger] {len(unfaithful_cases)} 条编造个案已入 faith_cases")

    # 3. 拒答:知识库没有的题
    abstain_lines: list[str] = []
    abstain_ok = 0
    for case in unanswerable:
        hits = await retrieve(
            session_factory=session_factory,
            query=case["query"],
            top_k=K10,
            strategy="hybrid_rerank",
            retriever=retriever,
            rewrite=args.rewrite,
        )
        verdict = await check_evidence(case["query"], format_evidence(build_citations(hits)))
        ok = not verdict.useful
        abstain_ok += 1 if ok else 0
        abstain_lines.append(
            f"- {'拒答 OK' if ok else '未拒答 !!'} {case['id']} {case['query']}"
            f"(证据 {len(hits)} 条,理由:{verdict.reason[:40] or '—'})"
        )
    if unanswerable:
        print(f"[abstain] {abstain_ok}/{len(unanswerable)}")

    # 4. 报告
    lines = ["# ch04 检索与生成评估", ""]
    lines.append(f"- 用例:{len(cases)}(可回答 {len(answerable)} / 不可回答 {len(unanswerable)})")
    lines.append(f"- 问法改写:{'开' if args.rewrite else '关'};重排:{'关' if args.no_rerank else '开'}")
    lines.append("")
    lines.append("## 检索(分策略 × 分桶)")
    lines.append("")
    header = "| 桶 | n | " + " | ".join(f"{s} R@5 / R@10 / MRR" for s in STRATEGIES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(STRATEGIES) + 2))
    for bucket in (BUCKETS if per_strategy else ()):
        n = len(per_strategy["dense"].get(bucket, []))
        if not n:
            continue
        cells = []
        for s in STRATEGIES:
            rs = per_strategy[s].get(bucket, [])
            cells.append(f"{_recall(rs, K5):.2f} / {_recall(rs, K10):.2f} / {_mrr(rs):.3f}")
        lines.append(f"| {bucket} | {n} | " + " | ".join(cells) + " |")
    cells = []
    for s in (STRATEGIES if per_strategy else ()):
        rs = [r for bucket in per_strategy[s].values() for r in bucket]
        cells.append(f"{_recall(rs, K5):.2f} / {_recall(rs, K10):.2f} / {_mrr(rs):.3f}")
    lines.append(f"| **全部** | {len(answerable)} | " + " | ".join(cells) + " |")
    lines.append("")
    if judged:
        lines.append("## 生成段")
        lines.append("")
        lines.append(f"- Faithfulness:{faith_pass}/{judged} = {faith_pass / judged:.1%}")
        lines.append("")
        lines.extend(gen_lines)
        lines.append("")
    if unanswerable:
        lines.append("## 拒答")
        lines.append("")
        lines.append(f"- 拒答正确:{abstain_ok}/{len(unanswerable)}")
        lines.append("")
        lines.extend(abstain_lines)
        lines.append("")

    report = "\n".join(lines)
    print()
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n报告已写入 {out}")

    await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ch04 检索评估(四策略对比 + 忠实度)")
    parser.add_argument("--cases", default="tests/eval/ch04_cases.jsonl")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--vector-store-uri", default=None)
    parser.add_argument("--rerank-model", default=None, help="重排模型路径(默认取配置 RERANK_MODEL)")
    parser.add_argument("--no-rerank", action="store_true", help="不加载重排模型")
    parser.add_argument("--no-generation", action="store_true", help="只算检索指标")
    parser.add_argument("--generation-only", action="store_true", help="只跑生成段(跳过四策略检索对比)")
    parser.add_argument("--no-ledger", action="store_true", help="不写 faith_cases 台账")
    parser.add_argument("--rewrite", action="store_true", help="打开 LLM 问法改写")
    parser.add_argument("--out", default=None, help="报告输出路径(markdown)")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
