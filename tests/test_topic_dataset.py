"""CH10 数据四步单测（清洗/归并/切分/增强）。"""

from __future__ import annotations

import json

from app.ml.topic.dataset import (
    augment_row,
    augment_train,
    clean_text,
    dedupe,
    fullwidth_to_halfwidth,
    map_rows,
    merge_labels,
    primary_label,
    scrub_pii,
    stratify_split,
    write_split,
)
from app.ml.topic.taxonomy import LABELS


# --- 第一步：清洗 -----------------------------------------------------------


def test_fullwidth_becomes_halfwidth():
    assert fullwidth_to_halfwidth("ＡＢＣ１２３") == "ABC123"
    assert fullwidth_to_halfwidth("能退吗？") == "能退吗？"  # 中文标点保留，语料要像真人打字
    assert fullwidth_to_halfwidth("订单　1003") == "订单 1003"


def test_scrub_pii_replaces_phone_email_order_and_id():
    text = "手机 13812345678 邮箱 a.b@shop.com 订单 20241015001 身份证 110101199001011234"
    out = scrub_pii(text)
    assert "13812345678" not in out and "<PHONE>" in out
    assert "a.b@shop.com" not in out and "<EMAIL>" in out
    assert "20241015001" not in out
    assert "110101199001011234" not in out and "<ID>" in out


def test_scrub_pii_keeps_short_order_numbers():
    # 池里的「订单 1003」是示意单号，四位不该被抹──抹了就看不出是哪单
    assert "1003" in scrub_pii("订单 1003 的包能退吗")


def test_clean_text_normalizes_whitespace_and_pii():
    assert clean_text("  订单　1003\n的包能退吗  ") == "订单 1003 的包能退吗"
    assert clean_text(None) == ""


def test_dedupe_keeps_first_occurrence():
    rows = [{"text": "能退吗", "labels": ["退换货"]}, {"text": "能退吗", "labels": ["尺码"]}]
    kept, dropped = dedupe(rows)
    assert [row["labels"] for row in kept] == [["退换货"]]
    assert len(dropped) == 1


# --- 第二步：术语表归并 -----------------------------------------------------


def test_merge_labels_drops_out_of_taxonomy_and_dedupes():
    kept, dropped = merge_labels(["退换货", "配送时效", "退换货"])
    assert kept == ["退换货"]
    assert dropped == ["配送时效"]


def test_merge_labels_falls_back_to_other():
    merged, _ = map_rows([{"text": "x", "labels": ["天气", "心情"]}])
    assert merged[0]["labels"] == ["其他"]


def test_map_rows_counts_dropped_labels():
    rows = [{"text": "a", "labels": ["按时送达"]}, {"text": "b", "labels": ["按时送达", "退货"]}]
    _, dropped = map_rows(rows)
    assert dropped["按时送达"] == 2


# --- 第三步：分层切分 -------------------------------------------------------


def _fake_rows() -> list[dict]:
    rows: list[dict] = []
    for idx, label in enumerate(["退换货"] * 20 + ["评价"] * 4 + ["价保"] * 30):
        rows.append({"text": f"{label}-{idx}", "labels": [label], "id": f"r{idx}"})
    return rows


def test_stratify_split_is_deterministic():
    rows = _fake_rows()
    first = stratify_split(rows, seed=7)
    second = stratify_split(rows, seed=7)
    assert [r["text"] for r in first["train"]] == [r["text"] for r in second["train"]]


def test_stratify_split_keeps_every_row_exactly_once():
    rows = _fake_rows()
    splits = stratify_split(rows)
    all_texts = [row["text"] for split in splits.values() for row in split]
    assert sorted(all_texts) == sorted(row["text"] for row in rows)


def test_stratify_split_puts_small_classes_into_val_and_test():
    splits = stratify_split(_fake_rows())
    for split in ("val", "test"):
        labels = {primary_label(row["labels"]) for row in splits[split]}
        # 「评价」只有 4 条，按纯比例会掉成 0 条 → 混淆矩阵那一类永远是空的
        assert "评价" in labels, f"{split} 里没有小类"


def test_stratify_split_respects_rough_ratios():
    splits = stratify_split(_fake_rows())
    total = sum(len(v) for v in splits.values())
    assert 0.7 <= len(splits["train"]) / total <= 0.88


# --- 第四步：增强（只扩训练集） ---------------------------------------------


def test_augment_row_keeps_labels_untouched():
    row = {"text": "这个鞋子能退货吗", "labels": ["退换货"]}
    new_row, how = augment_row(row, __import__("random").Random(1))
    assert new_row["labels"] == ["退换货"]
    assert how in {"synonym", "prefix", "suffix"}
    assert new_row["text"] != row["text"]


def test_augment_row_skips_overlong_text():
    row = {"text": "很" * 64, "labels": ["其他"]}
    new_row, how = augment_row(row, __import__("random").Random(1))
    assert how == "none" and new_row["text"] == row["text"]


def test_augment_train_keeps_originals_and_marks_copies():
    rows = [{"text": "这个包能退货吗", "labels": ["退换货"], "id": "t1"}]
    out, stats = augment_train(rows, copies=2, seed=1)
    assert rows[0] in out  # 原行一条不少
    made = [row for row in out if row.get("origin") == "augmented"]
    assert made and all(row["labels"] == ["退换货"] for row in made)
    assert sum(stats.values()) == 2  # 每条训练样本都尝试增强了一次


def test_augment_train_zero_copies_is_noop():
    rows = [{"text": "能换货吗", "labels": ["退换货"]}]
    out, _ = augment_train(rows, copies=0)
    assert out == rows


# --- 落盘 -------------------------------------------------------------------


def test_write_split_returns_sha256(tmp_path):
    path = tmp_path / "train.jsonl"
    meta = write_split(path, [{"text": "能退吗", "labels": ["退换货"]}])
    assert meta["rows"] == 1 and meta["labels"] == 1 and len(meta["sha256"]) == 64
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["text"] == "能退吗"


def test_taxonomy_still_seventeen():
    assert len(LABELS) == 17
