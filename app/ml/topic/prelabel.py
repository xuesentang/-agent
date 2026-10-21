"""LLM 预标 + 人工抽审（第十章需求 2）。

## 为什么要有这一步

需求原话是「大模型照术语表预标、人工抽审」。纯人工标 2600 条成本太高，纯模型标没人
敢信——折中做法是模型全量预标、人抽一部分核对，核出来的准确率决定预标结果能不能直接
当训练标签用。

## 两件事必须分开

- **预标**（`prelabel_texts`）：调模型，只吐标签，**不写库**。结果落 jsonl 文件，
  可追溯、可对比、可重跑。
- **回写**（`apply_review`）：只认人工填过的 `gold` 字段，算准不准，`--write-back`
  才写 `topic_classifications`。预标未核 → 永远不会被当成人工结论。

池子里 39 条真实问题全量预标一遍也才 39 次调用，没必要攒批。

## Prompt 从哪儿来

类目、边界说明、近邻口径全部 `taxonomy.py` 现取，不复制一份。预标 prompt 和训练时
标注的口径必须字面一致，否则模型学的和模型标的是两套概念。
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from app.ml.topic.taxonomy import NEIGHBOR_RULES, LABELS, taxonomy_prompt_block, validate_labels

PRELABLE_SYSTEM = """你是电商客服后台的工单分类员。只做一件事:把用户原话标进给定的类目表。

类目表(只能从这里选):
{taxonomy}

划界口径:
{rules}

输出要求:
- 只输出 JSON,不要任何其他内容;
- labels 是类目名数组,每个元素必须一字不差来自上面的类目表;
- 一句话里有几个诉求就打几个标签:"买大了想退"→ 尺码 + 退换货;
- 只能打一个就只打一个,不要凑数;
- 一个类目都对不上,或信息不足,打 ["其他"];
- 不要解释,不要自造类目。

输出 JSON 示例:{{"labels": ["尺码", "退换货"]}}"""

SYNTH_SYSTEM = """你在帮电商客服造训练语料。只造【{label}】这一类目的用户原话。

这一类管什么:{boundary}

要求:
- 每条一行,口语化,像真实买家打字,长度 6~25 字;
- 只涉及【{label}】,不要捎带其他类目的诉求(不许出现退款、补差价、补货这类别的类目的事);
- 句式要散:有的带商品名,有的带订单号,有的省略主语,有的带错别字;
- 不要编号,不要解释,不要引号,只输出 {count} 行原话。"""


class TopicLabels(BaseModel):
    """预标的结构化输出。字段名和 `topic_classifications` 的语义对齐。"""

    labels: list[str] = Field(description="命中的类目名,必须取自给定类目表,可多个")


def build_prelabel_messages(text: str) -> list[tuple[str, str]]:
    """一条问题的预标消息。返回 (role, content) 列表,便于单测断言口径。"""
    system = PRELABLE_SYSTEM.format(
        taxonomy=taxonomy_prompt_block(), rules=NEIGHBOR_RULES
    )
    return [("system", system), ("human", f"用户原话:{text}")]


def _coerce_labels(result) -> list[str]:
    """把结构化输出收敛成合法标签。

    供应商偶尔直接回 dict 而不是 pydantic 对象,这里两种都认;标签本身再过
    `validate_labels`——模型自造类目留着会污染分布页统计。
    """
    if isinstance(result, TopicLabels):
        raw = result.labels
    elif isinstance(result, dict):
        raw = result.get("labels") or []
    else:
        raw = getattr(result, "labels", []) or []
    return validate_labels(raw)


async def prelabel_one(model, text: str, *, method: str | None = None) -> list[str]:
    """标一条。失败不抛——批量跑的时候一条坏调用不该带塌整批。"""
    from app.config import get_settings

    structured = model.with_structured_output(
        TopicLabels, method=method or get_settings().llm_structured_method
    )
    messages = build_prelabel_messages(text)
    result = await structured.ainvoke(messages)
    return _coerce_labels(result)


async def prelabel_texts(
    texts: list[str],
    *,
    model=None,
    concurrency: int = 4,
    method: str | None = None,
) -> list[dict]:
    """批量预标,返回 [{text, labels, error}] 与输入等长且顺序一致。

    `concurrency` 别开太大:小模型供应商的限流是按并发算的,4 条并发的总耗时
    通常比 16 条打完再重试要短。
    """
    if model is None:
        from app.core.llm import get_extract_model

        model = get_extract_model()

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(text: str) -> dict:
        async with sem:
            try:
                return {"text": text, "labels": await prelabel_one(model, text, method=method), "error": None}
            except Exception as exc:  # noqa: BLE001 - 单条失败记下来继续
                return {"text": text, "labels": [], "error": f"{type(exc).__name__}: {exc}"}

    return await asyncio.gather(*(_one(t) for t in texts))


def build_synth_messages(label: str, count: int) -> list[tuple[str, str]]:
    """造数消息:一次一类,只造这一类的话。"""
    from app.ml.topic.taxonomy import BOUNDARIES

    system = SYNTH_SYSTEM.format(
        label=label, boundary=BOUNDARIES[label], count=count
    )
    return [("system", system), ("human", f"给我 {count} 条【{label}】的用户原话。")]


def parse_synth_lines(raw: str, label: str) -> list[dict]:
    """把模型吐的多行文本切成样本行。

    清洗在这里做第一遍:去编号前缀、去引号、去空白、丢过短的(「退款」两个字
    当训练样本等于噪声),同一批里重复的也丢掉。
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        # 去 1. / 1、/ - / * 这类编号前缀
        for prefix in ("-", "*", "•"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        head, sep, tail = text.partition(". ")
        if sep and head.isdigit():
            text = tail.strip()
        head, sep, tail = text.partition("、")
        if sep and head.isdigit():
            text = tail.strip()
        text = text.strip("“”\"'「」『』 ")
        if len(text) < 4 or text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "labels": [label], "source": "llm_synth"})
    return rows


async def synth_rows(model, *, per_label: int, labels=None, concurrency: int = 3) -> list[dict]:
    """按类目造带标签的句子,给 v2 数据集补量用。

    每类单独一次调用:一次让它造 17 类的话,模型会把类目边界搅在一起,回来还得人工
    拆,不如一类一问。造出来的行 `labels` 就是请求的那一类——prompt 明确禁止捎带
    别的类目,所以这里按单标签记。
    """
    targets = list(labels or LABELS)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(label: str) -> list[dict]:
        async with sem:
            messages = build_synth_messages(label, per_label)
            try:
                result = await model.ainvoke(messages)
                raw = getattr(result, "content", "") or ""
            except Exception:  # noqa: BLE001 - 某一类失败,别的类照造
                return []
            return parse_synth_lines(raw, label)

    groups = await asyncio.gather(*(_one(label) for label in targets))
    return [row for group in groups for row in group]


def sample_for_review(items: list[dict], n: int, *, seed: int = 42) -> list[dict]:
    """按固定种子抽样,保证「同一份预标结果抽出来的清单」可复现。

    抽不满就全给——39 条的量级抽 40 条时不该报错。
    """
    if n <= 0 or n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def render_review_markdown(items: list[dict], *, source: str = "") -> str:
    """人工抽审清单。空的那一列等人自己填,填完回 `--apply`。"""
    lines = [
        "# CH10 预标抽审清单",
        "",
        f"来源:{source}" if source else "",
        f"条数:{len(items)}",
        "",
        "改法:在「人工核定」列写类目名,多个用「、」隔开;认同预标就写 `ok`;",
        "拿不准就留空(留空的行不计入准确率)。填完存成同名 `.review.jsonl` 的 `gold` 字段。",
        "",
        "| # | 原话 | LLM 预标 | 人工核定 |",
        "|---|------|----------|----------|",
    ]
    for idx, item in enumerate(items, 1):
        text = str(item.get("text", "")).replace("|", "\\|").replace("\n", " ")
        labels = "、".join(item.get("labels") or []) or "(空)"
        lines.append(f"| {idx} | {text} | {labels} |  |")
    return "\n".join(line for line in lines if line != "") + "\n"


def parse_gold(raw) -> list[str] | None:
    """解析人工核定格。`ok` 表示认同预标(由调用方处理),空/None 表示没核。"""
    if raw is None:
        return None
    if isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        text = str(raw).strip()
        if not text:
            return None
        if text.lower() in {"ok", "对", "同意", "y", "yes"}:
            return ["__OK__"]
        for sep in ("、", ",", "，", "/", "|", " "):
            text = text.replace(sep, "\n")
        items = [seg for seg in text.splitlines() if seg.strip()]
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return cleaned or None


def score_against_gold(rows: list[dict]) -> dict:
    """预标 vs 人工核定的成绩单。

    两项指标说两件事:
    - `exact_match`:整条集合完全一致的比例,业务上「能不能直接用」看这个;
    - 微平均 P/R/F1:标签级别的召回与误标,漏标和错标的代价不一样时看这个。

    只统计 `gold` 非空的行;人工填 `ok` 的行按「预标即答案」计入。
    """
    total = 0
    exact = 0
    tp = fp = fn = 0
    mismatches: list[dict] = []

    for row in rows:
        gold_raw = parse_gold(row.get("gold"))
        if not gold_raw:
            continue
        pred = list(row.get("labels") or [])
        gold = pred if gold_raw == ["__OK__"] else validate_labels(gold_raw)
        total += 1
        if set(pred) == set(gold):
            exact += 1
        else:
            mismatches.append({"text": row.get("text"), "pred": pred, "gold": gold})
        for label in set(pred) | set(gold):
            in_pred, in_gold = label in pred, label in gold
            if in_pred and in_gold:
                tp += 1
            elif in_pred:
                fp += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reviewed": total,
        "exact": exact,
        "exact_match": round(exact / total, 4) if total else 0.0,
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
        "mismatches": mismatches,
    }


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
