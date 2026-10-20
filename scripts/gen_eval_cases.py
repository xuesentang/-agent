"""评估集出题器(E4):让模型在「只看目录、看不到答案」的条件下写用户问题。

为什么要这么绕:
    上一版评估集是我照着知识库写的——`expected_section` 直接抄章节名,题目用词也贴着
    文档用词。这等于把答案的措辞漏给检索,BM25 一撞就中,分数没有意义。
    这里把「出题」和「标注」拆开:出题者只拿到 文档标题 + 小节标题 + 型号清单
    (知道地图,不知道内容),标注(小节与要点)由脚本按骨架回填、我人工复核。

产出:`tests/eval/ch04_cases_draft.jsonl`,字段与最终评估集一致,
D 桶(near/far)不在本脚本里——那需要「库里没有什么」的知情人手工出,见 ch04_cases.jsonl。

用法:
    python scripts/gen_eval_cases.py                 # 生成草稿
    python scripts/gen_eval_cases.py --per-bucket 8 --out tests/eval/ch04_cases_draft.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kb_blueprint as blueprint
from expand_knowledge import _model

from app.rag.doc_loader import load_docs

DOC_DIR = Path("data/knowledge_docs")

# 三个原始文档的事实表(不在 blueprint 里,手工补,用于回填要点)
EXTRA_FACTS: dict[tuple[str, str], list[str]] = {
    ("退货退款政策", "政策总则"): ["7 日内可申请退货", "质量问题 15 日内", "退款按原支付路径返回"],
    ("退货退款政策", "运费规则"): ["满 99 元包邮", "未满 99 元收 8 元", "退货运费谁承担"],
    ("退货退款政策", "退款时效"): ["1 至 3 个工作日内发起退款", "银行卡 3 至 7 个工作日到账"],
    ("退货退款政策", "换货规则"): ["签收后 7 天内可申请换货", "3 个工作日内发出"],
    ("退货退款政策", "价保与补偿"): ["签收后 7 天价保", "1 个工作日内退回差价"],
    ("售后处理手册", "工单类型与时效"): ["退货退款 2 小时内响应", "维修 7 至 15 个工作日"],
    ("商品常见问题", "尺码与版型"): ["脚型偏宽建议选大半码", "部分热门款有宽楦版本"],
    ("商品常见问题", "库存与补货"): ["每周三补货", "到货提醒"],
    ("商品常见问题", "保修与质保"): ["12 个月内非人为损坏免费保修"],
}

BUCKET_SPEC = {
    "A_policy": {
        "n": 8,
        "how": "政策与流程的直问。句子完整、口语但信息齐全,像用户在 App 里打字提问。",
        "extra": "问题里不要出现「本公司」「依据政策第几条」这类书面语。",
    },
    "B_model": {
        "n": 8,
        "how": "必须点名具体型号(W20 / W40 / W60 / LP100 / LP200 / LP300 / TN2 / TN4 / SB200 / SB400 之一),问该型号的规格、差异或适用场景。",
        "extra": "型号写法可以贴近真实用户(如「w20」「LP 200」),但一条题里只用一种写法。",
    },
    "C_colloquial": {
        "n": 8,
        "how": "用别称、口语、行业黑话提问,禁止出现小节标题里的原词。例如「邮费」「运费」互替,「开票」「发票」互替。",
        "extra": "句子里可以有语气词和错别字。",
    },
    "E_multi": {
        "n": 6,
        "how": "一句话里问两件不同的事(两部分属于不同小节),用一个问句或两个短句连起来。",
        "extra": "两部分必须都能在目录里找到对应小节。",
    },
}

PROMPT = """你在为电商客服知识库的检索评测出题。你**只能看到文档目录,看不到文档内容**。

下面是知识库的目录(文档标题 → 小节标题):
{catalog}

商品型号清单:{models}

请出 {n} 道真实用户会问的问题,要求:
- {how}
- {extra}
- **不要把小节标题原样抄成问题**;要像用户自己在打字,而不是管理员在念目录;
- 每道题标注它应该命中的文档标题与小节标题(必须从上面目录里选,逐字一致);
- 如果是跨小节问题,主小节填 h2、另一部分填 h2_b。

只输出一个 JSON 对象(不要数组、不要解释),格式:
{{"cases": [{{"query": "问题原文", "doc": "文档标题", "h2": "小节标题", "h2_b": "另一小节或空", "difficulty": "easy|medium|hard"}}]}}"""

NOISE_PROMPT = """下面是一批电商客服问题。请为每一道改写出一条「更难检索」的变体,保持问的是同一件事:
- 加错别字或同音字(如「保修」写成「包修」、「运费」写成「云费」);
- 或者改型号写法(「W20」→「w-20」「W 20」);
- 或者去掉上下文只留半句(如「这个能退吗」);
- 或者加情绪化前缀(如「急!」「你们搞什么」);
每道题只选一种手法,并在 noise 字段写明用了哪种。

输入:
{cases}

只输出一个 JSON 对象(不要数组、不要解释),格式:
{{"cases": [{{"id": "原题 id", "query": "改写后的问题", "noise": ["typo"]}}]}}"""


def catalog() -> tuple[str, dict[str, list[str]]]:
    """从落盘文档读目录(标题 + H2),这才是库里真实存在的小节。"""
    lines: list[str] = []
    index: dict[str, list[str]] = {}
    for path in sorted(DOC_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = ""
        h2s: list[str] = []
        for ln in text.splitlines():
            if ln.startswith("# ") and not title:
                title = ln[2:].strip()
            elif ln.startswith("## "):
                h2s.append(ln[3:].strip())
        if title:
            index[title] = h2s
            lines.append(f"- {title}")
            for h in h2s:
                lines.append(f"    - {h}")
    return "\n".join(lines), index


def _json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def facts_for(doc_title: str, h2: str) -> list[str]:
    """按骨架回填要点;找不到就返回空(不编造)。"""
    key = (doc_title, h2)
    if key in EXTRA_FACTS:
        return EXTRA_FACTS[key]
    for d in blueprint.DOCS:
        if d["title"] != doc_title:
            continue
        for sec in d["sections"]:
            if sec["h2"] == h2:
                return sec["facts"][:3]
    return []


async def main_async(args) -> int:
    cat, index = catalog()
    models = " / ".join(
        ["W20", "W40", "W60", "S10", "S20", "T5", "LP100", "LP200", "LP300", "LP500", "TN2", "TN4", "SB200", "SB400"]
    )
    model = _model()

    raw: list[dict] = []
    for bucket, spec in BUCKET_SPEC.items():
        n = args.per_bucket or spec["n"]
        resp = await model.ainvoke(
            [
                (
                    "human",
                    PROMPT.format(
                        catalog=cat,
                        models=models,
                        n=n,
                        how=spec["how"],
                        extra=spec["extra"],
                    ),
                )
            ],
            response_format={"type": "json_object"},
        )
        items = _json_array(getattr(resp, "content", "") or "")
        print(f"[{bucket}] 模型给出 {len(items)} 条")
        for item in items:
            doc, h2 = (item.get("doc") or "").strip(), (item.get("h2") or "").strip()
            if doc not in index or h2 not in index[doc]:
                print(f"  跳过(目录对不上):{doc} / {h2}  ← {item.get('query')}")
                continue
            raw.append({**item, "bucket": bucket, "_doc": doc, "_h2": h2})

    # 回填 id / 要点 / 权重
    cases: list[dict] = []
    for i, item in enumerate(raw, start=1):
        points = facts_for(item["_doc"], item["_h2"])
        cases.append(
            {
                "id": f"{item['bucket'][0]}{i:02d}",
                "query": item["query"].strip(),
                "bucket": item["bucket"],
                "difficulty": item.get("difficulty") or "medium",
                # 用「文档标题 > 小节」而不是光写小节:旧版两份政策都有「版本说明」,
                # 只写小节会让两篇互相串标,命中判定虚高。
                "expected_section": f"{item['_doc']} > {item['_h2']}",
                "expected_keywords": points,
                "answerable": True,
                "source": "simulated",
                "noise": [],
                "weight": 2 if item["bucket"] in ("B_model", "C_colloquial") else 1,
                "note": f"生成自目录:{item['_doc']}",
            }
        )

    # 噪声变体
    payload = "\n".join(
        json.dumps({"id": c["id"], "query": c["query"]}, ensure_ascii=False) for c in cases
    )
    resp = await model.ainvoke(
        [("human", NOISE_PROMPT.format(cases=payload))], response_format={"type": "json_object"}
    )
    noisy = _json_array(getattr(resp, "content", "") or "")
    by_id = {c["id"]: c for c in cases}
    variants: list[dict] = []
    for item in noisy:
        base = by_id.get(item.get("id") or "")
        if not base or not item.get("query"):
            continue
        variants.append(
            {
                **base,
                "id": f"{base['id']}n",
                "query": str(item["query"]).strip(),
                "noise": item.get("noise") or ["rewrite"],
                "note": f"{base['note']};噪声变体",
            }
        )
    print(f"[noise] 生成 {len(variants)} 条变体")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for case in cases + variants:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"written {out}:{len(cases)} 干净 + {len(variants)} 噪声")

    no_points = sum(1 for c in cases if not c["expected_keywords"])
    if no_points:
        print(f"!! {no_points} 条没回填到要点,需要人工补(否则覆盖度算不准)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="生成评估集草稿(题目与标注分离)")
    parser.add_argument("--per-bucket", type=int, default=0, help="每桶题数(默认按 BUCKET_SPEC)")
    parser.add_argument("--out", default="tests/eval/ch04_cases_draft.jsonl")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
