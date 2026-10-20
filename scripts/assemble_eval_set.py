"""把出题器草稿收拾成可用的评估集(E4 收尾)。

这一步为什么必须有:
    `gen_eval_cases.py` 只看得到目录,看不到内容,所以它会把答案标到**旧版政策**上
    (「退换货政策(2024 版)」和现行「退货退款政策」在目录里长得一样),或者把睡袋
    SB200 标到「小家电型号总表」——它只能按字面猜。这类错标必须人工改,不能让脚本
    猜。噪声变体同理:模型把它们原样抄了一遍,只是贴了个 `typo` 标签,等于没做噪声。

    所以这里做三件事,全部显式记录:
    1. PATCH_SECTION:人工改对的小节标注(附原因);
    2. PATCH_QUERY:前提与库内事实冲突的题,改问句而不是改标注;
    3. 噪声变体改为**规则生成**——确定性、可复现、不花 token,而且规则改写更接近
       真人打字(漏标点、加错别字、把型号拆开写)。

产出:`tests/eval/ch04_cases.jsonl`(干净题 + 噪声变体 + D 桶负例)。

用法:
    python scripts/assemble_eval_set.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kb_blueprint as blueprint
from gen_eval_cases import EXTRA_FACTS

DRAFT = Path("tests/eval/ch04_cases_draft.jsonl")
REFUSE = Path("tests/eval/ch04_cases_refuse_v2.jsonl")
OUT = Path("tests/eval/ch04_cases.jsonl")

# 出题器把答案标到了旧版政策上:旧版是脏数据,现行版才是正解。
PATCH_SECTION: dict[str, tuple[str, str]] = {
    "A02": ("退货退款政策 > 运费规则", "问的是现行退货运费,旧版 129 元包邮是脏数据"),
    "A08": ("配送与运费规则 > 发货时效", "旧版 72 小时已作废,现行 48 小时"),
    "C18": ("退货退款政策 > 换货规则", "问换大一码谁付邮费,归换货规则而非运费规则"),
    "C20": ("配送与运费规则 > 发货时效", "旧版 72 小时已作废"),
    "E27": ("退货退款政策 > 运费规则", "旧版运费规则已作废"),
    "B15": ("型号对照总表 > 户外装备型号总表", "SB200/SB400 是睡袋,不是小家电"),
}

# 题面前提与库内事实冲突:改问句,不改标注。
PATCH_QUERY: dict[str, tuple[str, str]] = {
    "E25": (
        "我买的LP300咖啡机显示故障代码E3,这个代码是什么意思?要修的话得寄回去吗,多久能修好?",
        "原题问「上门维修」,库里维修流程是寄回、无上门,原题不可答",
    ),
    "E30": (
        "W10这个型号是不是已经停产了?停产了以后维修配件还买得到吗?",
        "原题问 SB200 停产,库里停产的是 W10 和 LP50,SB200 在售",
    ),
}

# 骨架里没有、但正文明确写了的事实,手工补(只补真正在正文里的)。
EXTRA_KEYWORDS: dict[str, list[str]] = {
    "A06": ["3 个工作日内发出换货商品", "换尺码优先同一仓库调拨", "换不同商品走退货退款再下单"],
}

# 原 D 桶的题被证明是「可答」的:题库里没有 ≠ 库里没有,这一条要靠人读库才能发现。
# 判据是 payment_policy 明写「不支持货到付款」——答「不支持」就是正确答案,不该要求拒答。
EXTRA_CLEAN: list[dict] = [
    {
        "id": "A31",
        "query": "你们支持货到付款吗",
        "bucket": "A_policy",
        "difficulty": "easy",
        "expected_section": "支付与优惠规则 > 支持的支付方式",
        "expected_keywords": ["不支持货到付款", "不支持对公转账", "客服无法为个别订单单独开通"],
        "answerable": True,
        "note": "原为 D01(far),核对库内发现 payment_policy 明确写了不支持货到付款,是判断题不是无解题",
    },
]

# 噪声规则:同一道题只施加一种手法,施加不了就顺延到下一种。
TYPO = {
    "运费": "云费",
    "退款": "退宽",
    "保修": "包修",
    "尺码": "尺玛",
    "包邮": "包游",
    "发票": "发漂",
}
MODEL_RE = re.compile(r"\b(W|LP|TN|SB)(\d{2,3})\b")


def facts_for(doc_title: str, h2: str) -> list[str]:
    if (doc_title, h2) in EXTRA_FACTS:
        return EXTRA_FACTS[(doc_title, h2)]
    for d in blueprint.DOCS:
        if d["title"] != doc_title:
            continue
        for sec in d["sections"]:
            if sec["h2"] == h2:
                return sec["facts"][:3]
    return []


def split_section(path: str) -> tuple[str, str]:
    head, _, tail = path.partition(" > ")
    return head.strip(), tail.strip()


def _model_format(query: str) -> str:
    m = MODEL_RE.search(query)
    assert m
    alt = f"{m.group(1).lower()}-{m.group(2)}"
    return query[: m.start()] + alt + query[m.end() :]


def _typo(query: str) -> str:
    for word, typo in TYPO.items():
        if word in query:
            return query.replace(word, typo, 1)
    return query


def _context_drop(query: str) -> str:
    cut = max(query.find("?"), query.find("？"))
    return query[: cut + 1]


def _emotion(query: str) -> str:
    return "急!" + re.sub(r"[?？]", "", query) + ",快回一下"


def _punctuation(query: str) -> str:
    return re.sub(r"[，。？！,?！]", "", query)


def noise_variants(query: str, index: int) -> tuple[str, str]:
    """返回 (变体, 手法)。同一道题只施加一种手法,按题号轮转,让五种手法都用上。

    为什么不按「优先级」挑:第一版按优先级挑,结果 30 条里 11 条是加情绪前缀,
    噪声维度退化成「每条题都加一句脏话」。轮转才能把 typo / 型号写法 / 掉上下文
    这几种在报告里分开看。
    """
    pool: list[tuple[str, callable]] = []
    if MODEL_RE.search(query):
        pool.append(("model_format", _model_format))
    if any(word in query for word in TYPO):
        pool.append(("typo", _typo))
    if query.count("?") + query.count("？") >= 2:
        pool.append(("context_drop", _context_drop))
    pool.append(("emotion", _emotion))
    pool.append(("punctuation", _punctuation))

    for offset in range(len(pool)):
        how, fn = pool[(index + offset) % len(pool)]
        text = fn(query)
        if text != query:
            return text, how
    return query, "none"


def main() -> int:
    clean = [
        json.loads(line)
        for line in DRAFT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not json.loads(line)["id"].endswith("n")
    ]

    for case in clean:
        cid = case["id"]
        if cid in PATCH_QUERY:
            case["query"], why = PATCH_QUERY[cid]
            case["note"] += f";改题:{why}"
        if cid in PATCH_SECTION:
            case["expected_section"], why = PATCH_SECTION[cid]
            case["note"] += f";改标:{why}"

    for case in clean:
        if case["id"] in EXTRA_KEYWORDS:
            case["expected_keywords"] = EXTRA_KEYWORDS[case["id"]]
        else:
            doc, h2 = split_section(case["expected_section"])
            points = facts_for(doc, h2)
            # 找不到就清空,让下面的「仍缺要点」报出来;留着旧要点等于悄悄用错标的要点。
            case["expected_keywords"] = points

    clean += [{**c, "source": "manual", "noise": [], "weight": 1} for c in EXTRA_CLEAN]

    variants: list[dict] = []
    for i, case in enumerate(clean):
        text, how = noise_variants(case["query"], i)
        variants.append(
            {
                **case,
                "id": f"{case['id']}n",
                "query": text,
                "noise": [how],
                "note": f"{case['note']};噪声变体({how})",
            }
        )

    refuse = [
        json.loads(line) for line in REFUSE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    rows = clean + variants + refuse
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    no_points = [c["id"] for c in clean if not c["expected_keywords"]]
    print(f"written {OUT}:干净 {len(clean)} + 噪声 {len(variants)} + D 桶 {len(refuse)} = {len(rows)}")
    print("噪声手法:" + "、".join(sorted({v["noise"][0] for v in variants})))
    if no_points:
        print(f"!! 仍缺要点:{no_points}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
