"""知识文档发现与元数据解析(build_knowledge 的输入侧,ch03)。

文档目录里的 .md 文件可带一行元数据头(HTML 注释):

    <!-- rag-doc: {"content_type": "policy", "category": "退换货"} -->

- content_type:faq / policy / manual,决定三格映射走哪条分支,缺省 policy;
- category:文档级分类(faq 分支用作 category,policy/manual 分支忽略,
  它们的 category 取上级标题路径,见 indexer.to_knowledge_chunks);
- JSON 坏损或 content_type 非法时回落默认值——不让一份坏文档卡住建库流水线;
- 非 .md 文件忽略;按文件名排序,保证建库顺序可复现。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_META_RE = re.compile(r"<!--\s*rag-doc:\s*(\{.*?\})\s*-->", re.DOTALL)
_CONTENT_TYPES = ("faq", "policy", "manual")


@dataclass(frozen=True)
class DocMeta:
    """文档头元数据。"""

    content_type: str = "policy"
    category: str = ""


def parse_doc_meta(text: str) -> DocMeta:
    """解析文档头元数据;缺失 / JSON 坏损 / 字段非法时回落 DocMeta() 默认值。"""
    match = _META_RE.search(text)
    if not match:
        return DocMeta()
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return DocMeta()
    if not isinstance(raw, dict):
        return DocMeta()
    content_type = str(raw.get("content_type", "")).strip()
    if content_type not in _CONTENT_TYPES:
        content_type = "policy"
    category = str(raw.get("category", "")).strip()
    return DocMeta(content_type=content_type, category=category)


def load_docs(source: Path) -> list[tuple[Path, str, str]]:
    """扫描文档源,返回 build_index 的 docs 入参:[(path, content_type, category)]。

    source 可以是目录(递归不进入子目录,.md 全收)或单个 .md 文件;
    路径不存在返回空列表(build_index 传空 docs = 纯补跑 pending 模式)。
    """
    source = Path(source)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".md")
    else:
        return []
    docs: list[tuple[Path, str, str]] = []
    for path in files:
        meta = parse_doc_meta(path.read_text(encoding="utf-8"))
        docs.append((path, meta.content_type, meta.category))
    return docs
