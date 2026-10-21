"""LLM 预标 + 人工抽审（第十章需求 2）。

    # 池里真实问题全量预标,顺手抽 10 条出人工清单(不写库)
    python scripts/prelabel_topics.py --review 10

    # 照术语表造数,给 v2 数据集补量(每类 20 条)
    python scripts/prelabel_topics.py --synth 20

    # 人工核完,把 gold 填进 review_*.jsonl,算预标准确率
    python scripts/prelabel_topics.py --apply reports/ch10_prelabel/review-20261021-1530.jsonl

    # 核过的结果同时写回 topic_classifications(核过才写)
    python scripts/prelabel_topics.py --apply <同上> --write-back

退出码:0 正常;1 调用全失败或文件读不到。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")

OUT_DIR = Path("reports/ch10_prelabel")
SYNTH_DIR = Path("data/topic_v2")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _fail_rate(rows: list[dict]) -> None:
    failed = [row for row in rows if row.get("error")]
    if failed:
        print(f"\n调用失败 {len(failed)} 条(已跳过,不写进清单):")
        for row in failed[:5]:
            print(f"  - {row['text'][:30]}: {row['error']}")


async def run_prelabel(args) -> int:
    from app.db import get_engine, get_sessionmaker
    from app.ml.topic.prelabel import (
        prelabel_texts,
        render_review_markdown,
        sample_for_review,
        write_jsonl,
    )
    from app.models import Base
    from app.repositories import topic_repo

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with get_sessionmaker()() as session:
        questions = await topic_repo.list_all(session, limit=args.limit)
    if not questions:
        print("池子里没有问题可预标")
        return 0

    print(f"取到 {len(questions)} 条问题,开始预标(并发 {args.concurrency})…")
    results = await prelabel_texts(
        [q.raw_question for q in questions], concurrency=args.concurrency
    )

    stamp = _stamp()
    rows = [
        {
            "question_id": question.id,
            "text": question.raw_question,
            "labels": result["labels"],
            "source": "pool",
            "error": result["error"],
        }
        for question, result in zip(questions, results)
    ]
    ok_rows = [row for row in rows if not row["error"]]

    jsonl_path = OUT_DIR / f"prelabel-{stamp}.jsonl"
    write_jsonl(jsonl_path, rows)  # 失败的也留着,error 字段说明原因

    if not ok_rows:
        print("全部调用失败,检查 .env 里的模型配置")
        return 1

    counts: dict[str, int] = {}
    for row in ok_rows:
        for label in row["labels"]:
            counts[label] = counts.get(label, 0) + 1
    multi = sum(1 for row in ok_rows if len(row["labels"]) > 1)

    print(f"\n预标 {len(ok_rows)} 条,多标签 {multi} 条")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<8}{count:>4}")
    _fail_rate(rows)
    print(f"\n明细:{jsonl_path}")

    if args.review > 0:
        picked = sample_for_review(ok_rows, args.review, seed=args.seed)
        # 清单里去掉 error/time,加一列空的 gold 等人工填
        review_rows = [
            {"question_id": row["question_id"], "text": row["text"], "labels": row["labels"], "gold": ""}
            for row in picked
        ]
        review_jsonl = OUT_DIR / f"review-{stamp}.jsonl"
        write_jsonl(review_jsonl, review_rows)
        md_path = OUT_DIR / f"review-{stamp}.md"
        md_path.write_text(
            render_review_markdown(review_rows, source=f"池内真实问题，种子 {args.seed}"),
            encoding="utf-8",
        )
        print(f"人工抽审 {len(review_rows)} 条:{md_path}")
        print(f"填 gold 的底稿:{review_jsonl}")
    return 0


async def run_synth(args) -> int:
    from app.core.llm import get_bulk_model
    from app.ml.topic.prelabel import synth_rows, write_jsonl

    model = get_bulk_model(max_tokens=2048)
    rows = await synth_rows(model, per_label=args.synth, concurrency=args.concurrency)
    if not rows:
        print("一条都没造出来,检查模型配置")
        return 1

    stamp = _stamp()
    path = SYNTH_DIR / f"synthetic-{stamp}.jsonl"
    write_jsonl(path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["labels"][0]] = counts.get(row["labels"][0], 0) + 1
    print(f"造了 {len(rows)} 条(目标每类 {args.synth} 条):{path}")
    for label, count in counts.items():
        print(f"  {label:<8}{count:>4}")
    return 0


async def run_apply(args) -> int:
    from app.ml.topic.prelabel import read_jsonl, score_against_gold
    from app.repositories import topic_repo

    path = Path(args.apply)
    if not path.exists():
        print(f"文件不存在:{path}")
        return 1
    rows = read_jsonl(path)
    if any(not row.get("gold") for row in rows):
        print(f"提示:{sum(1 for r in rows if not r.get('gold'))} 条没填 gold,不计入统计")

    score = score_against_gold(rows)
    if not score["reviewed"]:
        print("一条都没核,先填 gold 再来")
        return 1

    print(
        f"核了 {score['reviewed']} 条:完全一致 {score['exact']} 条"
        f"({score['exact_match'] * 100:.1f}%),"
        f"微 P/R/F1 = {score['micro_precision']:.4f}/{score['micro_recall']:.4f}/{score['micro_f1']:.4f}"
    )
    for item in score["mismatches"][:10]:
        print(f"  ✗ {item['text'][:26]}  预标={'、'.join(item['pred'])}  人工={'、'.join(item['gold'])}")

    report = OUT_DIR / f"score-{_stamp()}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# CH10 预标准确率（人工抽审结果）",
                "",
                f"- 抽审条数：{score['reviewed']}",
                f"- 完全一致：{score['exact']}（{score['exact_match'] * 100:.1f}%）",
                f"- 微平均 P/R/F1：{score['micro_precision']:.4f} / "
                f"{score['micro_recall']:.4f} / {score['micro_f1']:.4f}",
                "",
                "## 不一致的行",
                "",
                "| 原话 | 预标 | 人工核定 |",
                "|---|---|---|",
                *[
                    f"| {m['text']} | {'、'.join(m['pred'])} | {'、'.join(m['gold'])} |"
                    for m in score["mismatches"]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"报告:{report}")

    if args.write_back:
        from app.db import get_engine, get_sessionmaker
        from app.ml.topic.prelabel import parse_gold
        from app.models import Base
        from app.ml.topic.taxonomy import validate_labels

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        pairs = []
        for row in rows:
            gold = parse_gold(row.get("gold"))
            if not gold or not row.get("question_id"):
                continue
            labels = list(row["labels"]) if gold == ["__OK__"] else validate_labels(gold)
            pairs.append((int(row["question_id"]), labels))
        async with get_sessionmaker()() as session:
            written = await topic_repo.upsert_many(session, pairs)
        print(f"写回 {written} 条(人工核定优先于模型归类)")
    else:
        print("未写库(要写回加 --write-back)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="主题标签 LLM 预标 / 造数 / 抽审回写")
    parser.add_argument("--review", type=int, default=0, help="抽多少条进人工清单(默认 0)")
    parser.add_argument("--limit", type=int, default=None, help="池内最多预标多少条")
    parser.add_argument("--synth", type=int, default=0, metavar="N", help="每类造 N 条训练语料")
    parser.add_argument("--apply", default=None, metavar="JSONL", help="人工核过的 jsonl")
    parser.add_argument("--write-back", action="store_true", help="apply 时写回 topic_classifications")
    parser.add_argument("--seed", type=int, default=42, help="抽审种子")
    parser.add_argument("--concurrency", type=int, default=4, help="并发调用数")
    args = parser.parse_args()

    if args.apply:
        return asyncio.run(run_apply(args))
    if args.synth:
        return asyncio.run(run_synth(args))
    return asyncio.run(run_prelabel(args))


if __name__ == "__main__":
    raise SystemExit(main())
