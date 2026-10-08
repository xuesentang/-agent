from customer_service.knowledge_parser import parse_markdown
from customer_service.knowledge_parser import retire_stale_chunks


def test_heading_path_and_questions():
    chunks = parse_markdown("# 售后\n## 运费说明\n订单满 99 元包邮。", "shipping.md")
    assert len(chunks) == 1
    assert chunks[0].category == "售后"
    assert chunks[0].questions == "运费说明"
    assert chunks[0].answer == "订单满 99 元包邮。"
    assert chunks[0].section_path == "售后 / 运费说明"


def test_long_text_overlap_keeps_complete_sentences():
    text = "# 规则\n## 说明\n" + "第一句。第二句。第三句。第四句。"
    chunks = parse_markdown(text, "policy.md", max_chars=12, overlap_chars=4)
    assert len(chunks) >= 2
    assert all(chunk.answer.endswith("。") for chunk in chunks)
    assert "第三句。" in chunks[0].answer and "第三句。" in chunks[1].answer


def test_table_chunks_repeat_header():
    text = "# 售后\n## 费用表\n| 地区 | 运费 |\n| --- | --- |\n| 湖南 | 8元 |\n| 北京 | 12元 |"
    chunks = parse_markdown(text, "table.md", max_chars=46, overlap_chars=8)
    assert len(chunks) == 2
    assert all("| 地区 | 运费 |" in chunk.answer for chunk in chunks)
    assert "湖南" in chunks[0].answer and "北京" in chunks[1].answer


def test_noncontiguous_heading_levels_preserve_parent():
    chunks = parse_markdown("# 总则\n### 费用\n第一条。\n## 退货\n第二条。", "policy.md")
    assert [x.section_path for x in chunks] == ["总则 / 费用", "总则 / 退货"]


def test_long_sentence_splits_on_clause_boundary():
    chunks = parse_markdown("# 总则\n## 说明\n第一条说明很长，第二条说明也很长，第三条说明仍然很长。", "policy.md", max_chars=16, overlap_chars=4)
    assert len(chunks) >= 2
    assert all(x.answer.endswith(("，", "。")) for x in chunks)


def test_reimport_retire_removed_section(tmp_path):
    from sqlalchemy.orm import sessionmaker
    from customer_service.db import make_engine
    from customer_service.knowledge_store import KnowledgeStore
    from customer_service.models import Base

    engine = make_engine(f"sqlite:///{tmp_path / 'stale.db'}")
    Base.metadata.create_all(engine)
    store = KnowledgeStore(sessionmaker(engine, expire_on_commit=False))
    original = parse_markdown("# 政策\n## 旧运费\n旧说明。\n## 新运费\n新说明。", "policy.md")
    rows = store.add_chunks(original)
    for row in rows:
        store.mark_done(row.id, str(row.id))
    current = parse_markdown("# 政策\n## 新运费\n新说明。", "policy.md")
    store.add_chunks(current)
    retired = retire_stale_chunks(store, "policy.md", {chunk.source_key for chunk in current})
    assert retired == [rows[0].id]
    assert [x.answer for x in store.get_done([row.id for row in rows])] == ["新说明。"]
