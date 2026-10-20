"""把事实骨架扩写成知识文档(CH04 评测真实化 E2)。

输入:`scripts/kb_blueprint.py` 的 DOCS(型号/数值/版本冲突点由人写死)。
输出:`data/knowledge_docs/*.md`(与 build_knowledge 的输入格式一致)。

为什么不是「让模型写整篇」:
    陷阱必须跨文档一致——W20 无宽楦、W60 有 4E;旧版 15 天无理由、现行 7 天。
    模型自由发挥时这些点会漂,评估集的 ground truth 就标不出来。所以这里做的是
    「按骨架扩写」:事实由骨架给,模型只负责篇幅、语气、表格与啰嗦表述。

校验:
    每段生成后回检事实里的型号/数值 token 是否逐字出现,缺了就用「必须逐字出现」
    重试一次;仍缺则报错退出,不静默写入一份缺事实的文档。

用法:
    python scripts/expand_knowledge.py                      # 全部 12 篇
    python scripts/expand_knowledge.py --only shoes_manual.md legacy_return_policy_2024.md
    python scripts/expand_knowledge.py --force              # 覆盖已存在的文档
    python scripts/expand_knowledge.py --scale 1.5          # 目标字数放大 1.5 倍(扩库冲块数用)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kb_blueprint as blueprint

from app.core.llm import get_bulk_model

DEFAULT_TARGET = Path("data/knowledge_docs")
BATCH_CHARS = 1500  # 单次调用的目标字数上限:再大容易截断(见下面的 max_tokens 说明)
MAX_CONCURRENCY = 4
# 业务对话的 LLM_MAX_TOKENS 是 512,写文档远远不够(一千多字的正文就上千 token),
# 而且思考模式下推理 token 也算在内。这里单独开一个长输出客户端,不动全局配置。
GEN_MAX_TOKENS = 8192
GEN_TIMEOUT = 300

# 需要逐字出现的 token:型号(W20/LP300/TN4/SB200)、数值、百分比。
_TOKEN_RE = re.compile(r"[A-Z]{1,2}\d{1,3}|\d+(?:\.\d+)?")

KIND_GUIDE = {
    "prose": "写成条款式叙述段落,可用小标题分点,但不要写成问答。",
    "table": "至少包含一个 markdown 表格,把骨架里带数值的参数列成表;表格后补一段解释性文字。",
    "faq": "写成一问一答的 Q/A 形式,每个 Q 一行、A 一段,问题要像真实用户会问的口吻。",
}

SYSTEM = """你是电商品牌「{brand}」的知识库内容运营,负责写内部知识文档。

硬性要求:
1. 骨架里给出的事实(型号、数值、期限、费率、阈值)必须**逐字保留**,不得改写数字、不得四舍五入、不得添加骨架里没有的数值。
2. 可以补充解释、举例、免责声明、流程说明,但不得与骨架冲突;骨架没提到的事实不要编造具体数值。
3. 语气要像真实的电商政策文档:啰嗦、有重复表述、有「以页面说明为准」这类兜底句,不要写成简洁的标准答案。
4. 不要输出 H1 以外的元信息,不要写「本文由 AI 生成」,不要输出 markdown 代码块围栏。
5. 只输出指定小节的正文,每个小节以 `## 小节标题` 开头,顺序与给定顺序一致。{extra}"""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text))


def _missing(required: set[str], produced: str) -> list[str]:
    got = _tokens(produced)
    return sorted(t for t in required if t not in got)


def _batches(doc: dict, scale: float) -> list[list[dict]]:
    """把小节按目标字数装箱,单批不超过 BATCH_CHARS。"""
    out: list[list[dict]] = []
    cur: list[dict] = []
    acc = 0
    for sec in doc["sections"]:
        cost = int(sec["target"] * scale)
        if cur and acc + cost > BATCH_CHARS:
            out.append(cur)
            cur, acc = [], 0
        cur.append({**sec, "want": cost})
        acc += cost
    if cur:
        out.append(cur)
    return out


def _prompt(doc: dict, batch: list[dict], retry_missing: list[str] | None = None) -> str:
    lines = [
        f"文档标题:{doc['title']}",
        f"文档类型:{doc['content_type']}(policy=政策条款 / manual=产品手册 / faq=常见问题)",
    ]
    for sec in batch:
        lines.append("")
        lines.append(f"## {sec['h2']}")
        lines.append(f"写作要求:{KIND_GUIDE[sec['kind']]}")
        lines.append(f"目标字数:约 {sec['want']} 字(±20%)")
        lines.append("必须包含的事实:")
        for fact in sec["facts"]:
            lines.append(f"- {fact}")
    if retry_missing:
        lines.append("")
        lines.append("上一版漏掉了这些必须出现的内容,这次务必逐字写进去:")
        lines.append("、".join(retry_missing))
    return "\n".join(lines)


async def _expand_batch(model, doc: dict, batch: list[dict], sem: asyncio.Semaphore) -> str:
    required: set[str] = set()
    for sec in batch:
        for fact in sec["facts"]:
            required |= _tokens(fact)

    async with sem:
        for attempt in (1, 2):
            missing = None if attempt == 1 else _missing(required, text)
            started = time.monotonic()
            resp = await model.ainvoke(
                [
                    ("system", SYSTEM.format(brand=blueprint.BRAND, extra="")),
                    ("human", _prompt(doc, batch, missing)),
                ]
            )
            text = (getattr(resp, "content", "") or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
            miss = _missing(required, text)
            print(
                f"  [{doc['file']} / {batch[0]['h2']}] 第{attempt}次 "
                f"{time.monotonic() - started:.0f}s {len(text)}字 缺{len(miss)}项",
                flush=True,
            )
            if not miss:
                return text
    raise RuntimeError(
        f"{doc['file']} 的批次 {batch[0]['h2']} 两次生成都缺事实:{_missing(required, text)}"
    )


def _strip_h1(text: str) -> str:
    """丢掉批次输出里模型自补的 H1:文档标题由组装阶段统一写,否则会出现重复标题。"""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("# ")]
    return "\n".join(lines).strip()


async def _expand_doc(model, doc: dict, sem: asyncio.Semaphore, scale: float) -> str:
    batches = _batches(doc, scale)
    parts = await asyncio.gather(*(_expand_batch(model, doc, b, sem) for b in batches))
    body = "\n\n".join(_strip_h1(p) for p in parts)
    header = (
        "<!-- AI 合成数据:仅用于演示,非真实平台政策 -->\n"
        f'<!-- rag-doc: {{"content_type": "{doc["content_type"]}", '
        f'"category": "{doc["category"]}"}} -->\n\n'
        f"# {doc['title']}\n\n"
    )
    return header + body + "\n"


def _model():
    """长输出客户端。全局 LLM_MAX_TOKENS=512 是给对话用的,写文档会截断。"""
    return get_bulk_model(max_tokens=GEN_MAX_TOKENS, timeout=GEN_TIMEOUT)


async def main_async(args) -> int:
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    docs = [d for d in blueprint.DOCS if not args.only or d["file"] in args.only]
    missing = set(args.only or []) - {d["file"] for d in docs}
    if missing:
        print(f"未知文档:{sorted(missing)}", file=sys.stderr)
        return 2

    todo = []
    for doc in docs:
        path = target / doc["file"]
        if path.exists() and not args.force:
            print(f"skip {path}(已存在,--force 覆盖)")
            continue
        todo.append(doc)
    if not todo:
        print("没有需要生成的文档")
        return 0

    model = _model()
    sem = asyncio.Semaphore(args.concurrency)
    # 篇间并发 + 篇内并发,总并发由信号量卡死。逐篇串行时单批 1 分钟上下,
    # 84 个批次要跑一个多小时,实测撑不住;并发 6 是既不撞限流又够快的档位。
    results = await asyncio.gather(
        *(_expand_doc(model, d, sem, args.scale) for d in todo), return_exceptions=True
    )
    failed = 0
    for doc, res in zip(todo, results, strict=True):
        if isinstance(res, Exception):
            failed += 1
            print(f"FAILED {doc['file']}:{res}", file=sys.stderr, flush=True)
            continue
        path = target / doc["file"]
        path.write_text(res, encoding="utf-8")
        print(f"written {path}  {len(res)} 字", flush=True)
    total = sum(
        len((target / d["file"]).read_text(encoding="utf-8"))
        for d in docs
        if (target / d["file"]).exists()
    )
    print(f"done: {len(todo) - failed} written, {failed} failed; 该批文档合计 {total} 字")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="按事实骨架扩写知识文档(AI 合成)")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--only", nargs="*", default=None, help="只生成指定文件名")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文档")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--scale", type=float, default=1.0, help="目标字数倍数")
    sys.exit(asyncio.run(main_async(parser.parse_args())))


if __name__ == "__main__":
    main()
