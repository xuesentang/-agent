"""引用编号与证据组装(ch04 生成质量控制)。

约定:命中的每一条证据在 prompt 里都有一个编号 [n],n 就是下发给前端的那份
citation 列表的序号;答案里出现的 [n] 由前端渲染成可点角标,点开显示该块的
章节路径与原文。

为什么编号要跟「喂进去的顺序」一致,而不是跟「答案里引用的顺序」一致:
模型只会照抄它看到过的编号,检索侧顺序一变,角标就指错原文——错得还很隐蔽。
所以只有一处产生编号(本模块),prompt 与前端共用同一份列表。

首尾放置(需求 2):越相关的越靠两头,别按分数堆中间。中间位置在大模型注意力
分布上是低谷(lost in the middle),第 1 条和最后一条才是被真正读进去的位置。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """一条证据 = 一个引用编号。"""

    n: int
    chunk_id: int
    section_path: str
    question: str
    answer: str
    score: float

    def to_payload(self) -> dict:
        """下发给前端的最小集合(前端据此渲染角标与原文面板)。"""
        return {
            "n": self.n,
            "chunk_id": self.chunk_id,
            "section_path": self.section_path,
            "question": self.question,
            "answer": self.answer,
            "score": round(self.score, 6),
        }


def build_citations(hits) -> list[Citation]:
    """命中列表 → 带编号的证据列表(顺序即检索序,分数降序)。"""
    out: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        out.append(
            Citation(
                n=i,
                chunk_id=int(chunk.id),
                section_path=(getattr(chunk, "section_path", None) or "").strip(),
                question=(getattr(chunk, "questions", None) or "").strip(),
                answer=(getattr(chunk, "answer", None) or "").strip(),
                score=float(hit.score),
            )
        )
    return out


def order_for_prompt(citations: list[Citation]) -> list[Citation]:
    """首尾放置:最相关的放第一和最后,次相关的往里靠。

    例:1,2,3,4,5 → 1,3,5,4,2(编号不变,只调喂给模型的位置)。
    编号与 citation 列表绑定,所以这里绝不能重编号。
    """
    if len(citations) <= 2:
        return list(citations)
    head: deque[Citation] = deque()
    tail: deque[Citation] = deque()
    for i, c in enumerate(citations):
        (head if i % 2 == 0 else tail).append(c)
    return list(head) + list(reversed(tail))


def format_evidence(citations: list[Citation]) -> str:
    """证据块文本:给模型读的那份,[n] 必须与 citations 序号一致。"""
    lines: list[str] = []
    for c in order_for_prompt(citations):
        where = c.section_path or c.question or "知识库"
        lines.append(f"[{c.n}] 章节:{where}")
        if c.question and c.question != where:
            lines.append(f"    问法:{c.question}")
        lines.append(f"    内容:{c.answer}")
    return "\n".join(lines)
