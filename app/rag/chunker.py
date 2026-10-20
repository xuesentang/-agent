"""Markdown 结构感知切分引擎(ch03 spec D4,提示词第三章文档处理)。

四条要求逐条落地:
1. 按标题层级结构感知切分——标题栈维护 section_path,块不跨标题边界;
2. 超长内容递归切——先段落、段落仍超限再按句子;
3. 块间重叠且裁到最近句号——重叠区只取完整句子,不留半截话;
4. 大表格按行切、每块复制表头。

纯函数零依赖:split_document(markdown, config) -> list[RawChunk],
不碰 DB / Milvus / 嵌入,单测可全分支覆盖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 句末符:中英文 + 分号(分号在中文里常表句内停顿,但作为切分点至少保证不断句)
SENT_END_CHARS = "。！？!?；;"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ChunkConfig:
    """切分配置。max_chars 是块长硬上限(含重叠);overlap_ratio 控制块间重叠比例。"""

    max_chars: int = 400
    overlap_ratio: float = 0.15
    key_clause_markers: tuple[str, ...] = ("必须", "严禁", "禁止", "重要", "注意")


@dataclass(frozen=True)
class RawChunk:
    """结构切分的原始产物:section_path + 正文 + 两个元数据标记。

    category/questions/answer 三格映射由调用方按 content_type 做
    (政策手册 questions=末级标题、category=上级路径;FAQ questions=真实问法),
    切分引擎只管结构与边界。
    """

    section_path: str
    heading: str
    body: str
    is_key_clause: bool
    is_table: bool = False


# ---------------------------------------------------------------- 句子工具


def _trim_to_sentence(text: str, keep_tail: bool) -> str:
    """把 text 裁到最近句号边界。keep_tail=True 取尾部约 n 字内首个句末之后的内容。

    找不到完整句子时返回 ""——宁可不重叠,也不留半截话(提示词硬要求)。
    """

    def first_sentence_end(s: str) -> int:
        for i, ch in enumerate(s):
            if ch in SENT_END_CHARS:
                return i
        return -1

    if keep_tail:
        idx = first_sentence_end(text)
        return text[idx + 1 :].strip() if idx >= 0 else ""
    # keep_tail=False:从末尾往前找最后一个句末符,截到其后
    last = -1
    for i, ch in enumerate(text):
        if ch in SENT_END_CHARS:
            last = i
    return text[: last + 1] if last >= 0 else text


def _sentence_tail(text: str, n: int) -> str:
    """取 text 末尾至多 n 字,向左裁到最近句末(含),只保留完整句子序列。"""
    if len(text) <= n:
        return text
    cut = text[-n:]
    return _trim_to_sentence(cut, keep_tail=True)


def _split_sentences(text: str) -> list[str]:
    """按句末符切句(保留句末符);无句末符的尾巴整体作为一句。"""
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_END_CHARS:
            sentences.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        tail = text[start:]
        if tail.strip():
            sentences.append(tail)
    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------- 块级切分


def _split_long_text(text: str, budget: int) -> list[str]:
    """超长文本递归切:先段落,单段仍超限再按句子,产物每块 <= budget。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(para) > budget:
            if cur:
                pieces.append(cur)
                cur = ""
            # 句子贪心装箱
            sent_buf = ""
            for sent in _split_sentences(para):
                if sent_buf and len(sent_buf) + len(sent) > budget:
                    pieces.append(sent_buf)
                    sent_buf = sent
                else:
                    sent_buf += sent
            if sent_buf:
                pieces.append(sent_buf)
        elif not cur:
            cur = para
        elif len(cur) + 2 + len(para) <= budget:
            cur = f"{cur}\n\n{para}"
        else:
            pieces.append(cur)
            cur = para
    if cur:
        pieces.append(cur)
    return pieces


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    """相邻块加重叠:后块头部拼上前块尾部(完整句子序列),总长不超 max_chars。

    切分时每块预算已预留 overlap 位(max_chars - overlap),故拼接后不超限。
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i > 0:
            tail = _sentence_tail(pieces[i - 1], overlap)
            if tail:
                piece = f"{tail}\n\n{piece}"
        out.append(piece)
    return out


def _split_table(lines: list[str], config: ChunkConfig) -> list[str]:
    """大表格按行切,每块复制表头(表头行 + 分隔行);短表格整块返回。"""
    header = lines[:2]
    data = lines[2:]
    whole = "\n".join(lines)
    if len(whole) <= config.max_chars:
        return [whole]
    chunks: list[str] = []
    cur = list(header)
    for row in data:
        candidate = cur + [row]
        if len("\n".join(candidate)) > config.max_chars and len(cur) > len(header):
            chunks.append("\n".join(cur))
            cur = list(header)
        cur.append(row)
    if len(cur) > len(header):
        chunks.append("\n".join(cur))
    return chunks


# ---------------------------------------------------------------- 文档解析


def _strip_comments(markdown: str) -> str:
    """剥掉 HTML 注释(如文档头的 rag-doc 元数据行)——注释不是知识内容。

    多行注释整体去掉;行内注释只去掉注释段,同行其余文字保留。
    """
    return _COMMENT_RE.sub("", markdown)


def _parse_blocks(markdown: str):
    """把 markdown 切成有序块序列:[("heading", level, title) | ("text", str) | ("table", [lines])]。

    文本块吞掉空行(空行只是段落分隔,不是块边界),遇标题 / 表格才断开,
    保证同一节里的连续段落进同一文本块、共享块间重叠。
    """
    blocks: list[tuple] = []
    text_buf: list[str] = []
    table_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            text = "\n".join(text_buf).strip()
            if text:
                blocks.append(("text", text))
            text_buf.clear()

    def flush_table() -> None:
        if table_buf:
            blocks.append(("table", list(table_buf)))
            table_buf.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            flush_text()
            flush_table()
            blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
            continue
        if line.lstrip().startswith("|"):
            flush_text()
            table_buf.append(line.strip())
            continue
        if not line.strip():
            # 空行:文本块内保留为段落分隔(压成一行哨兵,join 时还原)
            if text_buf:
                text_buf.append("")
            continue
        flush_table()
        text_buf.append(line.strip())
    flush_text()
    flush_table()
    return blocks


def split_document(markdown: str, config: ChunkConfig | None = None) -> list[RawChunk]:
    """结构感知切分入口:markdown 原文 → 有序 RawChunk 列表(文档序)。"""
    cfg = config or ChunkConfig()
    if not markdown or not markdown.strip():
        return []
    markdown = _strip_comments(markdown)

    overlap = int(cfg.max_chars * cfg.overlap_ratio)
    budget = cfg.max_chars - overlap

    stack: list[tuple[int, str]] = []  # (level, title)
    chunks: list[RawChunk] = []
    # 当前标题下的内容块(文本/表格交替),切分时保持文档序
    pending: list[tuple[str, object]] = []

    def current_path() -> str:
        return " > ".join(t for _, t in stack)

    def current_heading() -> str:
        return stack[-1][1] if stack else ""

    def flush_section() -> None:
        if not pending:
            return
        section_text = "\n\n".join(
            b[1] if b[0] == "text" else "\n".join(b[1]) for b in pending  # type: ignore[arg-type]
        )
        is_key = any(m in section_text for m in cfg.key_clause_markers)
        path, heading = current_path(), current_heading()

        # 每个文本块独立切分 + 加重叠(重叠只发生在本块内相邻块之间);
        # 表格块独立按行切;最终按文档序交叉归位。
        out: list[RawChunk] = []
        for kind, payload in pending:
            if kind == "text":
                for piece in _with_overlap(
                    _split_long_text(payload, budget), overlap  # type: ignore[arg-type]
                ):
                    out.append(RawChunk(path, heading, piece, is_key, is_table=False))
            else:
                for table_text in _split_table(payload, cfg):  # type: ignore[arg-type]
                    out.append(RawChunk(path, heading, table_text, is_key, is_table=True))
        chunks.extend(out)
        pending.clear()

    for block in _parse_blocks(markdown):
        if block[0] == "heading":
            flush_section()
            level, title = block[1], block[2]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            pending.append((block[0], block[1]))
    flush_section()
    return chunks
