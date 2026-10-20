"""引用组装单测(ch04):编号连续性、首尾放置、证据文本、下发载荷。"""

from app.rag.citations import Citation, build_citations, format_evidence, order_for_prompt


class FakeChunk:
    def __init__(self, cid, section_path="", questions="", answer=""):
        self.id = cid
        self.section_path = section_path
        self.questions = questions
        self.answer = answer

    def embedding_text(self):
        return f"{self.section_path}\n{self.questions}\n{self.answer}"


class FakeHit:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


def _citations(n: int) -> list[Citation]:
    hits = [
        FakeHit(FakeChunk(i, f"章{i}", f"问{i}", f"答{i}"), score=1.0 - i / 100)
        for i in range(1, n + 1)
    ]
    return build_citations(hits)


def test_build_citations_numbers_from_one_in_score_order():
    citations = _citations(3)
    assert [c.n for c in citations] == [1, 2, 3]
    assert [c.chunk_id for c in citations] == [1, 2, 3]
    assert citations[0].score > citations[1].score


def test_build_citations_tolerates_missing_fields():
    """挖出来的 QA 可能没有 section_path:回落到问法,不能拼出 "None"。"""
    citations = build_citations([FakeHit(FakeChunk(9, None, "怎么退货", "7 天内"), 0.5)])
    assert citations[0].section_path == ""
    assert citations[0].question == "怎么退货"


def test_order_for_prompt_puts_best_at_head_and_tail():
    """lost in the middle:第 1 名放首位、第 2 名放末位,中间按名次往里收。"""
    citations = _citations(5)
    ordered = order_for_prompt(citations)
    assert [c.n for c in ordered] == [1, 3, 5, 4, 2]
    assert ordered[0].n == 1 and ordered[-1].n == 2


def test_order_for_prompt_keeps_numbering():
    """位置换了,编号不能变——编号是前端回原文的唯一线索。"""
    ordered = order_for_prompt(_citations(4))
    assert sorted(c.n for c in ordered) == [1, 2, 3, 4]


def test_order_for_prompt_short_lists_unchanged():
    assert [c.n for c in order_for_prompt(_citations(2))] == [1, 2]


def test_format_evidence_carries_numbers_and_sections():
    text = format_evidence(_citations(2))
    assert "[1] 章节:章1" in text
    assert "内容:答2" in text
    # 首尾放置生效:第 1 条在最前
    assert text.index("[1]") < text.index("[2]")


def test_to_payload_is_json_friendly():
    payload = _citations(1)[0].to_payload()
    assert payload["n"] == 1 and payload["chunk_id"] == 1
    assert isinstance(payload["score"], float)
    assert set(payload) == {"n", "chunk_id", "section_path", "question", "answer", "score"}
