"""数据四步（第十章需求 1）：清洗 → 术语表归并 → 分层切分 → 只增强训练集。

## 这四步为什么不能省

- **清洗**：客服原话里混着手机号、订单号、邮箱。这些直接进训练集有两个后果——
  模型学会「认数字」而不是「认诉求」，而且一份带 PII 的训练集是不能外传的。
- **归并**：语料来自 legacy，标签有越界的（「物流慢」被标成「配送时效」）。越界标签
  不归并进 17 类，训练时会被静默丢弃，最后表现成「某一类召回怎么都上不去」。
- **分层切分**：随机切会让「评价」这种小类可能一条都不进验证集，评估矩阵就成空的。
- **只增强训练集**：验证/测试集一旦被增强，指标就变成「模型见过的东西考自己」，
  数字好看但没有意义。这条是硬线。

## 和已交付模型的关系

产出是 **v2**，只用来证明四步跑得通、以及给下一轮重训备料。已交付的 0.9687 来自
v1 冻结集，两者不要混着引用——README 与 dev-notes 里都写清楚这件事。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from app.ml.topic.taxonomy import LABELS

# --- 第一步：清洗 -----------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
# 订单号：带「订单/单号」前缀的、或裸的 8 位以上数字串。
# 门槛定在 6~8 位是有意的：真实订单号都很长，而语料里“订单 1003”这类四位示意单号
# 抹掉后会让「订单 1003 的包能退吗」和「订单 1005 的包能退吗」去重成一条。
_ORDER_PREFIXED = re.compile(r"((?:订单|单号|编号)\s*[:：]?\s*[A-Za-z]?)\d{6,}")
_BARE_LONG_DIGITS = re.compile(r"(?<!\d)\d{8,}(?!\d)")


def fullwidth_to_halfwidth(text: str) -> str:
    """全角字母/数字转半角，全角空格转半角空格。

    只转这三类，**中文标点一律保留**（，！？：、）。中文客服语料本来就该长这样，
    把「，」压成「,」看着像机翻，训练出来的模型对真实输入的分布也更远。
    真正需要统一的是数字和字母：全角「１」和半角「1」在分词器里是两个不同字符，
    不统一就是白送噪声。
    """
    out: list[str] = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(char)
    return "".join(out)


def scrub_pii(text: str) -> str:
    """抹掉手机号/邮箱/身份证/订单号，换成占位符。

    顺序有讲究：先抹长的（身份证 18 位），再抹手机号，最后才上「8 位以上裸数字」
    这条兜底——反过来的话，11 位手机号会先被当成订单号，日志里就看不出本该是手机号。
    """
    text = _EMAIL.sub("<EMAIL>", text)
    text = _IDCARD.sub("<ID>", text)
    text = _PHONE.sub("<PHONE>", text)
    text = _ORDER_PREFIXED.sub(r"\1<ORDER>", text)
    return _BARE_LONG_DIGITS.sub("<ORDER>", text)


def clean_text(text: str) -> str:
    """清洗单条文本。空白先压平，再去 PII，最后再压一次（占位符替换后可能出现双空格）。"""
    flat = " ".join(str(text or "").split())
    flat = fullwidth_to_halfwidth(flat)
    flat = scrub_pii(flat)
    return " ".join(flat.split()).strip()


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """按清洗后的文本去重，保留先出现的。返回 (保留, 丢弃)。"""
    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        key = row["text"]
        if key in seen:
            dropped.append(row)
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


# --- 第二步：术语表归并 -----------------------------------------------------


def merge_labels(labels) -> tuple[list[str], list[str]]:
    """把标签收敛到 17 类。返回 (合法标签, 被丢弃的越界标签)。

    全被丢光 → `["其他"]`，与 `taxonomy.validate_labels` 一个口径：业务上不允许空标签。
    被丢了什么要能看见，所以额外把丢弃项返回给调用方统计。
    """
    kept: list[str] = []
    dropped: list[str] = []
    for label in labels or []:
        text = str(label).strip()
        if text in LABELS:
            if text not in kept:
                kept.append(text)
        elif text:
            dropped.append(text)
    if not kept:
        kept = ["其他"]
    return kept, dropped


def map_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    """对全量行做归并，返回 (处理后的行, 丢弃标签计数)。"""
    dropped_counter: Counter = Counter()
    out: list[dict] = []
    for row in rows:
        labels, dropped = merge_labels(row.get("labels"))
        dropped_counter.update(dropped)
        out.append({**row, "labels": labels})
    return out, dropped_counter


# --- 第三步：分层切分 -------------------------------------------------------


def primary_label(labels: list[str]) -> str:
    """分层用的「主标签」= 标签列表的第一个。

    多标签行按第一个标签分层，是个近似：好处是简单可复现，坏处是分层后各类样本数
    不完全按标签数均衡。够用——这一步要的是「小类不进空集」，不是严格配额。
    """
    return labels[0] if labels else "其他"


def stratify_split(
    rows: list[dict], *, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> dict[str, list[dict]]:
    """按主标签分层切 80/10/10。同一个 seed 切出同一份数据。

    每类先各分一条给 val/test（该类的样本数 ≥3 时），剩下的按比例分。
    不这么做的话，样本数只有 4 条的小类按比例算下来 val/test 都是 0 条，
    混淆矩阵里那一类永远是空的，等于没评估。
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(primary_label(row["labels"]), []).append(row)

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label in sorted(groups):
        bucket = list(groups[label])
        rng.shuffle(bucket)
        size = len(bucket)
        if size >= 3:
            n_val = max(1, round(size * ratios[1]))
            n_test = max(1, round(size * ratios[2]))
            # 小类可能 1+1 超出剩余量，压回来保证训练集至少留 1 条
            while n_val + n_test >= size:
                if n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        else:
            n_val = n_test = 0
        out["val"].extend(bucket[:n_val])
        out["test"].extend(bucket[n_val : n_val + n_test])
        out["train"].extend(bucket[n_val + n_test :])

    for split in out:
        out[split].sort(key=lambda row: row["text"])
    return out


# --- 第四步：数据增强（只扩训练集） -----------------------------------------

# 同义替换：只换说法不换语义，且换完仍是同一类目
SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("退货", "退掉"),
    ("退款", "退钱"),
    ("运费", "邮费"),
    ("快递", "包裹"),
    ("优惠券", "券"),
    ("什么时候", "啥时候"),
    ("尺码", "码数"),
    ("发票", "票据"),
    ("订单", "单子"),
    ("保修", "质保"),
)

PREFIXES: tuple[str, ...] = ("想问下,", "请问,", "麻烦问一下,", "咨询一下,")
SUFFIXES: tuple[str, ...] = ("，谢谢", "，麻烦看下", "，急", "，谢谢啦")


def augment_row(row: dict, rng: random.Random) -> tuple[dict, str]:
    """生成一条增强副本，返回 (新行, 用了哪种手法)。

    三种手法各用一次，不叠加：叠加出来的句子越长越像机翻，训练时会教模型
    「长句 = 某个类」，反而学出捷径。换不动（同义词不在句子里）就换手法。
    """
    text = row["text"]
    candidates: list[tuple[str, str]] = []

    for source, target in SYNONYM_PAIRS:
        if source in text:
            candidates.append((text.replace(source, target, 1), "synonym"))
            break  # 一次只换一处，换多了语义容易漂

    if len(text) + 5 <= 64:  # 模型最大长度 64，超了会被截断,增强就白做了
        candidates.append((rng.choice(PREFIXES) + text, "prefix"))
        candidates.append((text + rng.choice(SUFFIXES), "suffix"))

    candidates = [(t, how) for t, how in candidates if t != text]
    if not candidates:
        return dict(row), "none"
    new_text, how = rng.choice(candidates)
    return {**row, "text": new_text, "origin": "augmented", "aug": how}, how


def augment_train(
    rows: list[dict], *, copies: int = 1, seed: int = 42
) -> tuple[list[dict], Counter]:
    """只扩训练集。返回 (原行 + 增强行, 手法计数)。"""
    rng = random.Random(seed)
    made: list[dict] = []
    stats: Counter = Counter()
    for row in rows:
        for _ in range(max(0, copies)):
            new_row, how = augment_row(row, rng)
            stats[how] += 1
            if how != "none":
                made.append({**new_row, "id": f"{row.get('id', 'row')}-aug{len(made)}"})
    return list(rows) + made, stats


# --- 落盘 -------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "rows": len(rows),
        "labels": sum(len(row["labels"]) for row in rows),
        "sha256": sha256_of(path),
    }


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
