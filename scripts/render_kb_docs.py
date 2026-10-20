"""把短事实渲染成知识文档(E2 另一半)。

为什么分两步:
    「模型写正文」单批要 60-90 秒,而且上游一并发就挂(见 dev-notes/ch04.md);
    「模型写 9 条一句话事实」两三秒就回来。所以篇幅交给模板,事实交给模型:
    事实文件是可提交的小 JSON,渲染是确定性的,重跑不花 token。

输入:`data/kb_facts/*.json`(mine_kb_facts.py 的产出)
输出:`data/knowledge_docs/*.md`(build_knowledge.py 的输入)

渲染风格刻意啰嗦:每节开头一句总述,条款式罗列,末尾一段「以页面为准」的兜底。
真实电商政策文档就是这样,过短过整齐反而让 BM25 太好命中。

用法:
    python scripts/render_kb_docs.py                  # 渲染全部
    python scripts/render_kb_docs.py --force          # 覆盖已存在
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FACTS_DIR = Path("data/kb_facts")
TARGET_DIR = Path("data/knowledge_docs")

LEADS = [
    "本节说明{sec}相关的具体规定，请在下单或使用前仔细阅读。",
    "以下为{sec}的处理口径，如与商品详情页、结算页展示不一致，以页面说明为准。",
    "{sec}按下列条款执行，未尽事宜按平台通用规则处理。",
]
TAILS = [
    "上述内容如有调整，将在本页面更新，更新后的内容适用于其后生成的订单。",
    "本节不构成对时效与金额的承诺，具体以订单实际履约情况与页面展示为准。",
    "如对上述规定存在疑问，建议在下单前通过在线客服确认，避免因理解偏差产生争议。",
    "因地区政策、承运方运力等客观因素导致的实际差异，本店将尽力协调但无法保证完全一致。",
]
PREFIXES = [
    "关于{kw}，",
    "针对{kw}，",
    "{kw}方面，",
    "需要说明的是，{kw}，",
]


def _kw(fact: str) -> str:
    """从事实里取一个短语当条目引子:优先逗号前部分,否则前 10 字。"""
    head = fact.split("，")[0].split("。")[0].strip()
    return head[:12] if len(head) <= 12 else head[:12]


def render_section(sec: dict, idx: int) -> str:
    h2 = sec["h2"]
    facts = sec["facts"]
    blocks = [f"## {h2}", "", LEADS[idx % len(LEADS)].format(sec=h2), ""]
    for i, fact in enumerate(facts, start=1):
        prefix = PREFIXES[i % len(PREFIXES)].format(kw=_kw(fact))
        blocks.append(f"{i}. {prefix}{fact}。")
    blocks.append("")
    blocks.append(TAILS[idx % len(TAILS)])
    return "\n".join(blocks)


def render_doc(payload: dict) -> str:
    header = (
        "<!-- AI 合成数据:仅用于演示,非真实平台政策 -->\n"
        f'<!-- rag-doc: {{"content_type": "{payload["content_type"]}", '
        f'"category": "{payload["category"]}"}} -->\n\n'
        f"# {payload['title']}\n\n"
    )
    body = "\n\n".join(render_section(s, i) for i, s in enumerate(payload["sections"]))
    return header + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="事实 JSON → 知识文档 markdown")
    parser.add_argument("--facts-dir", default=str(FACTS_DIR))
    parser.add_argument("--target", default=str(TARGET_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    facts_dir, target = Path(args.facts_dir), Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(facts_dir.glob("*.json"))
    if not files:
        print(f"没有事实文件:{facts_dir}(先跑 scripts/mine_kb_facts.py)")
        sys.exit(1)

    written = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out = target / payload["file"]
        if out.exists() and not args.force:
            print(f"skip {out}")
            continue
        text = render_doc(payload)
        out.write_text(text, encoding="utf-8")
        written += 1
        print(f"written {out}({len(text)} 字符,{len(payload['sections'])} 节)")
    total = sum(len(p.read_text(encoding='utf-8')) for p in target.glob("*.md"))
    print(f"done: {written} 篇;目录合计 {total} 字符")


if __name__ == "__main__":
    main()
