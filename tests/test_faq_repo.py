"""faq_repo TDD(spec Review Focus 4:LIKE 转义)。"""

import pytest

from app.models.faq import Faq
from app.repositories import faq_repo


async def _seed(session) -> None:
    rows = [
        Faq(question="退货政策是什么", answer="签收后7天内可退", category="退换货"),
        Faq(question="运费怎么算", answer="满99元包邮,否则收8元运费", category="运费"),
    ]
    session.add_all(rows)
    await session.commit()


async def test_match_on_question(session):
    await _seed(session)
    hits = await faq_repo.search_by_keyword(session, "退货")
    assert len(hits) == 1
    assert hits[0].category == "退换货"


async def test_match_on_answer(session):
    await _seed(session)
    hits = await faq_repo.search_by_keyword(session, "包邮")
    assert [h.question for h in hits] == ["运费怎么算"]


async def test_no_match_returns_empty(session):
    await _seed(session)
    # 「邮费」不出现在种子数据里(只有「运费」)——ch02 预期漏召回样例。
    assert await faq_repo.search_by_keyword(session, "邮费") == []


async def test_like_wildcards_are_escaped(session):
    await _seed(session)
    # 未转义时 % 会匹配全部;转义后按字面匹配,数据里没有字面 % → 0 命中。
    assert await faq_repo.search_by_keyword(session, "%") == []
    assert await faq_repo.search_by_keyword(session, "_") == []
    # 反斜杠是 ESCAPE 字符本身,字面反斜杠也要转义(review Minor 2)。
    assert await faq_repo.search_by_keyword(session, "\\") == []


async def test_limit_caps_results(session):
    await _seed(session)
    # 空关键词 escape 后 pattern="%%" 匹配所有——但 limit=1 截断。
    hits = await faq_repo.search_by_keyword(session, "", limit=1)
    assert len(hits) == 1
