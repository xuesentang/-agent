"""CH10 LLM 预标 + 抽审单测。

不打真模型:预标的核心风险在于「模型吐脏标签怎么办」「人工核结果怎么算账」,
这两件事都能用假模型覆盖。真模型的预标质量走 `scripts/prelabel_topics.py` 实跑。
"""

from __future__ import annotations

import json

import pytest

from app.ml.topic.prelabel import (
    TopicLabels,
    build_prelabel_messages,
    build_synth_messages,
    parse_gold,
    parse_synth_lines,
    prelabel_texts,
    render_review_markdown,
    sample_for_review,
    score_against_gold,
    write_jsonl,
)
from app.ml.topic.taxonomy import LABELS, NEIGHBOR_RULES


class FakeStructured:
    """假的结构化输出链:按文本查表回标签,查不到就抛(模拟上游抽风)。"""

    def __init__(self, mapping: dict[str, list[str]], boom: set[str] | None = None):
        self.mapping = mapping
        self.boom = boom or set()
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        text = messages[-1][1].split(":", 1)[1].strip()
        if text in self.boom:
            raise RuntimeError("upstream 500")
        return TopicLabels(labels=self.mapping.get(text, []))


class FakeModel:
    def __init__(self, structured):
        self.structured = structured

    def with_structured_output(self, schema, method=None):
        self.method = method
        return self.structured


def test_prompt_carries_taxonomy_and_neighbor_rules():
    system = build_prelabel_messages("买大了想退")[0][1]
    assert "尺码" in system and "保修维修" in system
    assert NEIGHBOR_RULES in system
    assert "不要自造类目" in system


@pytest.mark.asyncio
async def test_prelabel_maps_labels_and_keeps_order():
    structured = FakeStructured({"买大了想退": ["尺码", "退换货"]})
    rows = await prelabel_texts(["买大了想退"], model=FakeModel(structured))
    assert rows[0]["labels"] == ["尺码", "退换货"]
    assert rows[0]["error"] is None


@pytest.mark.asyncio
async def test_prelabel_drops_illegal_labels_and_falls_back_to_other():
    structured = FakeStructured({"今天天气不错": ["天气", "心情"]})
    rows = await prelabel_texts(["今天天气不错"], model=FakeModel(structured))
    # 全被丢光 → 「其他」,分布页才不会出现空标签黑洞
    assert rows[0]["labels"] == ["其他"]


@pytest.mark.asyncio
async def test_prelabel_one_failure_does_not_kill_batch():
    structured = FakeStructured({"退货要多久到账": ["退换货"]}, boom={"发票怎么开"})
    rows = await prelabel_texts(
        ["退货要多久到账", "发票怎么开"], model=FakeModel(structured), concurrency=2
    )
    assert rows[0]["labels"] == ["退换货"]
    assert rows[1]["labels"] == [] and "upstream 500" in rows[1]["error"]


def test_sample_for_review_is_deterministic_and_capped():
    items = [{"text": f"问题{i}"} for i in range(20)]
    first = sample_for_review(items, 5, seed=7)
    assert first == sample_for_review(items, 5, seed=7)
    assert len(first) == 5
    assert sample_for_review(items, 99, seed=7) == items  # 抽不满给全量


def test_render_review_markdown_has_empty_gold_column():
    md = render_review_markdown([{"text": "买大了想退|急", "labels": ["尺码", "退换货"]}])
    row = [line for line in md.splitlines() if line.startswith("| 1 |")][0]
    assert "买大了想退\\|急" in row  # 竖线转义,否则表格错列
    assert "尺码、退换货" in row
    assert row.endswith("|  |")  # 人工核定一列留空


def test_parse_gold_variants():
    assert parse_gold("") is None
    assert parse_gold(None) is None
    assert parse_gold("ok") == ["__OK__"]
    assert parse_gold("尺码、退换货") == ["尺码", "退换货"]
    assert parse_gold(["尺码"]) == ["尺码"]


def test_score_counts_exact_and_micro_f1():
    rows = [
        {"text": "a", "labels": ["退换货", "尺码"], "gold": "尺码、退换货"},  # 顺序无关,算全对
        {"text": "b", "labels": ["运费"], "gold": "运费"},  # ok
        {"text": "c", "labels": ["退换货"], "gold": "退换货、运费"},  # 漏标运费
        {"text": "d", "labels": ["发票"], "gold": ""},  # 没核,不计
        {"text": "e", "labels": ["价保"], "gold": "ok"},  # 认同预标
    ]
    score = score_against_gold(rows)
    assert score["reviewed"] == 4
    assert score["exact"] == 3
    assert score["micro_precision"] == 1.0
    assert score["micro_recall"] == round(5 / 6, 4)  # 6 个金标,命中 5(漏了 c 的运费)
    assert [m["text"] for m in score["mismatches"]] == ["c"]


def test_parse_synth_lines_strips_numbering_and_dedupes():
    raw = "1. 买的鞋小了半码\n2、这个包能退吗\n- 耳机戴着夹耳\n买的鞋小了半码\n退款\n"
    rows = parse_synth_lines(raw, "尺码")
    assert [row["text"] for row in rows] == ["买的鞋小了半码", "这个包能退吗", "耳机戴着夹耳"]
    assert all(row["labels"] == ["尺码"] for row in rows)


def test_synth_prompt_forbids_neighbor_topics():
    system = build_synth_messages("尺码", 5)[0][1]
    assert "尺码" in system and "5" in system
    assert "不要捎带其他类目" in system


def test_write_jsonl_roundtrip(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = [{"text": "买大了", "labels": ["尺码"], "gold": ""}]
    assert write_jsonl(path, rows) == 1
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == rows[0]


def test_taxonomy_has_seventeen_labels():
    assert len(LABELS) == 17 and len(set(LABELS)) == 17
