"""评估集自检(E4 配套)。跑在评估之前,把「标签对不上库」这种错挡在跑批之前。

为什么需要:
    评估集里 `expected_section` 写错一个字,那一题就永远命中不了,指标会低得莫名
    其妙,而且看起来像「检索不行」。v1 就是靠人肉盯,21 块库盯得住,350 块盯不住。

用法:
    python -m tests.eval.validate_rag_cases --cases tests/eval/ch04_cases.jsonl \
        --database-url "sqlite+aiosqlite:///./data/devstack.db"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi", "D")
SUBTYPES = ("far", "near")
REQUIRED = (
    "id",
    "query",
    "bucket",
    "answerable",
    "expected_section",
    "expected_keywords",
    "source",
    "noise",
    "weight",
)


async def known_sections(database_url: str) -> set[str]:
    engine = create_async_engine(database_url)
    async with engine.connect() as conn:
        rows = (await conn.execute(text("SELECT DISTINCT section_path FROM knowledge_chunks"))).all()
    await engine.dispose()
    return {r[0] or "" for r in rows}


def check(cases: list[dict], sections: set[str]) -> list[str]:
    errors: list[str] = []
    ids = Counter(c["id"] for c in cases)
    errors += [f"重复 id:{i}" for i, n in ids.items() if n > 1]

    for case in cases:
        cid = case.get("id", "?")
        missing = [k for k in REQUIRED if k not in case]
        if missing:
            errors.append(f"{cid} 缺字段:{missing}")
            continue
        if case["bucket"] not in BUCKETS:
            errors.append(f"{cid} bucket 非法:{case['bucket']}")
        if not isinstance(case["weight"], int) or case["weight"] < 1:
            errors.append(f"{cid} weight 必须是 >=1 的整数")

        if case["bucket"] == "D":
            if case["answerable"]:
                errors.append(f"{cid} D 桶必须 answerable=false")
            if case.get("subtype") not in SUBTYPES:
                errors.append(f"{cid} D 桶 subtype 必须是 far/near")
            if case["expected_section"] or case["expected_keywords"]:
                errors.append(f"{cid} D 桶不得带 ground truth")
        else:
            if not case["answerable"]:
                errors.append(f"{cid} 非 D 桶必须 answerable=true")
            sec = case["expected_section"]
            if not sec:
                errors.append(f"{cid} 缺 expected_section")
            elif not any(sec in s for s in sections):
                errors.append(f"{cid} expected_section 在库里找不到:{sec!r}")
            if not case["expected_keywords"]:
                errors.append(f"{cid} 缺 expected_keywords(覆盖度算不出来)")

    # 噪声变体必须有原题
    by_id = {c["id"]: c for c in cases}
    for case in cases:
        if case["noise"] and not case["id"].endswith("n"):
            continue
        if case["id"].endswith("n"):
            base = case["id"][:-1]
            if base not in by_id:
                errors.append(f"{case['id']} 是噪声变体但找不到原题 {base}")
            elif case["expected_section"] != by_id[base]["expected_section"]:
                errors.append(f"{case['id']} 的 expected_section 与原题不一致")
    return errors


async def run(args) -> int:
    path = Path(args.cases)
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sections = await known_sections(args.database_url or get_settings().database_url)
    errors = check(cases, sections)

    buckets = Counter(c["bucket"] for c in cases)
    clean = sum(1 for c in cases if not c["noise"])
    noisy = len(cases) - clean
    sub = Counter(c.get("subtype") for c in cases if c["bucket"] == "D")
    print(f"用例 {len(cases)} 条:干净 {clean} / 噪声 {noisy}")
    print("桶:" + "、".join(f"{b}={buckets.get(b, 0)}" for b in BUCKETS))
    print("D 桶:" + "、".join(f"{k}={v}" for k, v in sorted(sub.items())))
    print(f"库内小节 {len(sections)} 个")
    if errors:
        print(f"\n!! {len(errors)} 个问题:")
        for e in errors[:40]:
            print(f"  - {e}")
        return 1
    print("自检通过")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="评估集自检")
    parser.add_argument("--cases", default="tests/eval/ch04_cases.jsonl")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
