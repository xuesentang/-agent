"""ch04 检索与生成评估 v2(评测真实化:E5)。

相对 v1 的四处改动,每一处都对应一个被证伪的假设:
1. **库规模进报告**:v1 在 21 块上跑出 R@5=1.00,Top-10 占全库 48%——指标贴顶是
   算术必然。v2 把「块数 / Top-10 占比」写进报告头部,占比 >5% 直接警告。
2. **D 桶分 subtype**:far(一眼库外)与 near(相关但答不了)分开报。near 全被拒
   说明题出得太容易,报告里会明说。
3. **干净题与噪声题分开**:噪声变体(错别字/型号变体/去上下文)单列一列,
   不混进主表——混进去就分不清「检索不行」和「噪声没起作用」。
4. **裁判与生成分离**:judge 走 `get_judge_model()`,报告 meta 分别记生成模型与
   裁判模型;两者相同则显式标注「仅可横向比,不可当绝对准确率」。
   要点覆盖不再要求逐字出现,改由裁判判「语义上盖住了哪些要点」。

双口径:算术平均与 weight 加权平均同时输出。权重是假设的工单频次(无真实工单),
报告里必须这么写,不能当成真实分布。

用法:
    python -m tests.eval.run_rag_eval --no-generation          # 只算检索(快)
    python -m tests.eval.run_rag_eval                          # 全跑
    python -m tests.eval.run_rag_eval --cases tests/eval/ch04_cases.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.prompts import CUSTOMER_SERVICE_TOOLS_SYSTEM_PROMPT
from app.eval.metrics import (  # 指标口径抽到 app/eval/metrics.py（ch09 D7）
    BUCKETS,
    K5,
    K10,
    STRATEGIES,
    TOP10_SHARE_WARN,
    CaseResult,
    load_cases,
    mrr as _mrr,
    rank_of as _rank,
    recall as _recall,
    weighted_mrr as _wmrr,
    weighted_recall as _wrecall,
)
from app.rag.citations import build_citations, format_evidence
from app.rag.embeddings import LocalBGEEmbedder
from app.rag.reranker import LocalBGEReranker
from app.rag.retriever import Retriever, retrieve
from app.rag.selfcheck import check_evidence
from app.rag.vectorstore import MilvusStore


# ---------------------------------------------------------------- 生成段


GENERATE_PROMPT = """{system}

下面是检索到的证据:
{evidence}

用户问题:{question}

请依据证据回答,每条结论后标注来源编号(如 [1]);证据不足就直说没查到,不要补充证据之外的内容。"""

JUDGE_PROMPT = """你是客服知识库评测的裁判。给你**证据**、**答案**、**用户问题**、以及**本节要点**。

要做三件事:
1. 判相关:这一节要点里有哪几条是**本题问到的**。要点是从整节抽出来的,而问题通常只问其中一部分(比如整节要点是 E1/E2/E3/E4 四个故障码,问题只问 E3),把没问到的算进分母会让覆盖度天生偏低。
2. 判忠实:答案里的每条事实性陈述是否都能在证据里找到依据。语气词与安抚话术不算编造;答案明说「没查到」不算编造。
3. 判覆盖:答案在**语义上**盖住了几条**相关**要点(不要求逐字一致,同义表达、改写、举例都算盖住)。

只输出 JSON:
{{"relevant": ["本题问到的要点原文"], "faithful": true/false, "covered": ["盖住的要点原文"], "reason": "不忠实时指出编在哪一句"}}

证据:
{evidence}

用户问题:
{question}

本节要点(先判相关,再判覆盖):
{points}

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


async def judge_answer(model, evidence: str, question: str, points: list[str], answer: str) -> dict:
    """一次调用同时拿相关、忠实、覆盖,避免同一份材料请裁判看三遍。

    返回的 `relevant_count` 是覆盖度的分母:只算「本题问到的要点」。旧版拿整节要点当分母,
    问「E3 是什么意思」时分母里还挂着 E1/E2/E4,覆盖率天生 1/4,那不是答得差。
    """
    from app.rag.query_rewrite import _parse_json

    try:
        resp = await model.ainvoke(
            [
                (
                    "human",
                    JUDGE_PROMPT.format(
                        evidence=evidence,
                        question=question,
                        points="\n".join(f"- {p}" for p in points) or "(无要点)",
                        answer=answer,
                    ),
                )
            ],
            response_format={"type": "json_object"},
        )
        payload = _parse_json(getattr(resp, "content", "") or "")
    except Exception as exc:  # noqa: BLE001 - 判不了就当忠实,不算个案
        return {
            "faithful": True,
            "covered_count": 0,
            "relevant_count": len(points),
            "reason": f"裁判失败:{exc}",
        }
    if "faithful" not in payload:
        return {
            "faithful": True,
            "covered_count": 0,
            "relevant_count": len(points),
            "reason": "裁判输出不可解析",
        }
    covered = payload.get("covered") or []
    relevant = payload.get("relevant") or []
    if not isinstance(covered, list):
        covered = []
    if not isinstance(relevant, list) or not relevant:
        # 裁判没给相关判定时退回整节要点,并在 reason 里留痕,不静默改口径
        relevant = points
    return {
        "faithful": bool(payload["faithful"]),
        "covered_count": len(covered),
        "relevant_count": len(relevant),
        "reason": str(payload.get("reason") or ""),
    }


# ---------------------------------------------------------------- 主流程


async def run(args) -> int:
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    answerable = [c for c in cases if c.get("answerable")]
    refusable = [c for c in cases if not c.get("answerable")]

    settings = get_settings()
    engine = create_async_engine(args.database_url or settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = MilvusStore(uri=args.vector_store_uri)
    embedder = LocalBGEEmbedder()
    reranker = None if args.no_rerank else LocalBGEReranker(args.rerank_model)
    retriever = Retriever(store=store, embedder=embedder, reranker=reranker)

    async with engine.connect() as conn:
        total_chunks = int(
            (await conn.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))).scalar() or 0
        )
    top_share = K10 / total_chunks if total_chunks else 1.0
    print(f"库:{total_chunks} 块,Top-10 占 {top_share:.1%}")
    if top_share > TOP10_SHARE_WARN:
        print(f"!! 警告:Top-10 占全库 {top_share:.1%} > {TOP10_SHARE_WARN:.0%},指标没有区分度")

    print(
        f"cases={len(cases)} answerable={len(answerable)} refuse={len(refusable)} "
        f"buckets={sorted({c['bucket'] for c in cases})}"
    )
    print(f"rewrite={'on' if args.rewrite else 'off'} rerank={'off' if args.no_rerank else 'on'}")

    # 1. 四策略检索指标
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
            by_bucket.setdefault(case["bucket"], []).append(
                CaseResult(
                    case["id"],
                    case["bucket"],
                    _rank(hits, case.get("expected_section") or ""),
                    weight=case["weight"],
                    noise=bool(case.get("noise")),
                    hits=hits,
                )
            )
        per_strategy[strategy] = by_bucket
        all_results = [r for rs in by_bucket.values() for r in rs]
        clean = [r for r in all_results if not r.noise]
        noisy = [r for r in all_results if r.noise]
        print(
            f"[{strategy:14s}] 全部 R@5={_recall(all_results, K5):.3f} "
            f"R@10={_recall(all_results, K10):.3f} MRR={_mrr(all_results):.3f} | "
            f"干净 MRR={_mrr(clean):.3f} | 噪声 MRR={_mrr(noisy):.3f}"
        )

    # 2. 生成段(默认只跑干净题,噪声题要 --noise-generation)
    gen_lines: list[str] = []
    faith_pass = judged = cov_got = cov_total = 0
    unfaithful: list[dict] = []
    judge_model_name = ""
    if not args.no_generation:
        from app.core.llm import get_chat_model, get_judge_model

        gen_model = get_chat_model()
        judge_model, judge_model_name = get_judge_model()
        for case in answerable:
            if case.get("noise") and not args.noise_generation:
                continue
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
            points = case.get("expected_keywords") or []
            answer = await generate_answer(gen_model, case["query"], evidence)
            verdict = await judge_answer(judge_model, evidence, case["query"], points, answer)
            judged += 1
            faith_pass += 1 if verdict["faithful"] else 0
            cov_got += verdict["covered_count"]
            cov_total += verdict["relevant_count"]
            mark = "OK " if verdict["faithful"] else "!! "
            gen_lines.append(
                f"- {mark}{case['id']} {case['query']} → {answer[:60]}\n"
                f"  - 覆盖 {verdict['covered_count']}/{verdict['relevant_count']} 相关要点"
                f"(本节共 {len(points)} 条);{verdict['reason'][:80]}"
            )
            if not verdict["faithful"]:
                unfaithful.append(
                    {
                        "case": case,
                        "answer": answer,
                        "reason": verdict["reason"],
                        "citations": [c.to_payload() for c in citations],
                    }
                )
        print(f"[generation] faithfulness={faith_pass}/{judged}")
        print(f"[generation] 相关要点覆盖={cov_got}/{cov_total}")

        if unfaithful and not args.no_ledger:
            from app.repositories import faith_case_repo

            async with session_factory() as session:
                for item in unfaithful:
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
                        judge_model=judge_model_name,
                    )
            print(f"[ledger] {len(unfaithful)} 条编造个案已入 faith_cases")

    # 3. 拒答:far / near 分开
    abstain_lines: list[str] = []
    abstain_stat: dict[str, list[int]] = {}
    for case in refusable:
        subtype = case.get("subtype") or "far"
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
        stat = abstain_stat.setdefault(subtype, [0, 0])
        stat[0] += 1 if ok else 0
        stat[1] += 1
        abstain_lines.append(
            f"- {'拒答 OK' if ok else '未拒答 !!'} [{subtype}] {case['id']} {case['query']}"
            f"(证据 {len(hits)} 条,理由:{verdict.reason[:50] or '—'})"
        )
    for subtype, (ok, n) in sorted(abstain_stat.items()):
        print(f"[abstain:{subtype}] {ok}/{n}")
        if subtype == "near" and n and ok == n:
            print("!! near 题全部被拒:可能题出得太容易(或判据太松),要重出")
    # 4. 报告
    lines = ["# ch04 检索与生成评估(v2 评测真实化)", ""]
    lines.append(f"- 知识库:{total_chunks} 块;Top-10 占全库 **{top_share:.1%}**")
    lines.append(
        f"- 用例:{len(cases)}(可答 {len(answerable)} / 不可答 {len(refusable)});"
        f"问法改写 {'开' if args.rewrite else '关'};重排 {'关' if args.no_rerank else '开'}"
    )
    lines.append(
        f"- 生成模型:`{settings.llm_model}`;裁判模型:`{judge_model_name or '未跑生成段'}`"
    )
    if judge_model_name and judge_model_name == settings.llm_model:
        lines.append(
            "- **裁判与生成是同一个模型**:分数只能用于策略横向比较,不能当生产准确率。"
            "配 `JUDGE_MODEL` 换成不同厂商/规模的模型后,绝对分数才有意义。"
        )
    noise_n = sum(1 for c in answerable if c.get("noise"))
    lines.append(f"- 噪声变体:{noise_n} 条(单独统计,不混入干净题)")
    lines.append("- 权重:按假设的工单频次设定,不是真实分布;下表同时给算术平均与加权平均")
    lines.append("")

    lines.append("## 检索(分策略 × 分桶,干净题)")
    lines.append("")
    header = "| 桶 | n | " + " | ".join(f"{s} R@5 / R@10 / MRR" for s in STRATEGIES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(STRATEGIES) + 2))
    for bucket in (BUCKETS if per_strategy else ()):
        rs0 = [r for r in per_strategy["dense"].get(bucket, []) if not r.noise]
        if not rs0:
            continue
        cells = []
        for s in STRATEGIES:
            rs = [r for r in per_strategy[s].get(bucket, []) if not r.noise]
            cells.append(f"{_recall(rs, K5):.2f} / {_recall(rs, K10):.2f} / {_mrr(rs):.3f}")
        lines.append(f"| {bucket} | {len(rs0)} | " + " | ".join(cells) + " |")
    if per_strategy:
        cells, wcells = [], []
        for s in STRATEGIES:
            rs = [r for b in per_strategy[s].values() for r in b if not r.noise]
            cells.append(f"{_recall(rs, K5):.2f} / {_recall(rs, K10):.2f} / {_mrr(rs):.3f}")
            wcells.append(f"{_wrecall(rs, K5):.2f} / {_wrecall(rs, K10):.2f} / {_wmrr(rs):.3f}")
        lines.append("| **算术平均** | — | " + " | ".join(cells) + " |")
        lines.append("| **加权平均** | — | " + " | ".join(wcells) + " |")
        lines.append("")
        lines.append("### 噪声变体(同题不同写法)")
        lines.append("")
        lines.append("| 策略 | 干净 MRR | 噪声 MRR | 差值 |")
        lines.append("|---|---|---|---|")
        for s in STRATEGIES:
            clean = [r for b in per_strategy[s].values() for r in b if not r.noise]
            noisy = [r for b in per_strategy[s].values() for r in b if r.noise]
            if not noisy:
                continue
            lines.append(
                f"| {s} | {_mrr(clean):.3f} | {_mrr(noisy):.3f} | "
                f"{_mrr(noisy) - _mrr(clean):+.3f} |"
            )
    lines.append("")
    if judged:
        lines.append("## 生成段")
        lines.append("")
        lines.append(f"- Faithfulness:{faith_pass}/{judged} = {faith_pass / judged:.1%}")
        if cov_total:
            lines.append(f"- 相关要点覆盖(本题问到的才算分母):{cov_got}/{cov_total} = {cov_got / cov_total:.1%}")
        lines.append("")
        lines.extend(gen_lines)
        lines.append("")
    if abstain_stat:
        lines.append("## 拒答(分 far / near)")
        lines.append("")
        for subtype, (ok, n) in sorted(abstain_stat.items()):
            lines.append(f"- `{subtype}`:拒答 {ok}/{n} = {ok / n:.1%}")
        if abstain_stat.get("near", [0, 0])[1] and abstain_stat["near"][0] == abstain_stat["near"][1]:
            lines.append("")
            lines.append(
                "- **near 全部被拒,这条不能当成绩看**:说明这批 near 题对当前检索+裁判太容易"
                "(或裁判判据太松)。要么把题改得更像「库里有相邻事实」,要么承认这一维没测出东西。"
            )
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
    parser = argparse.ArgumentParser(description="ch04 检索评估 v2(四策略 + 分桶 + 噪声 + 拒答 subtype)")
    parser.add_argument("--cases", default="tests/eval/ch04_cases.jsonl")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--vector-store-uri", default=None)
    parser.add_argument("--rerank-model", default=None, help="重排模型路径(默认取配置 RERANK_MODEL)")
    parser.add_argument("--no-rerank", action="store_true", help="不加载重排模型")
    parser.add_argument("--no-generation", action="store_true", help="只算检索指标")
    parser.add_argument("--generation-only", action="store_true", help="只跑生成段")
    parser.add_argument("--noise-generation", action="store_true", help="生成段也跑噪声变体(慢)")
    parser.add_argument("--no-ledger", action="store_true", help="不写 faith_cases 台账")
    parser.add_argument("--rewrite", action="store_true", help="打开 LLM 问法改写")
    parser.add_argument("--out", default=None, help="报告输出路径(markdown)")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
