"""doc_loader 单测:元数据解析 + 文档发现。"""

from app.rag.doc_loader import load_docs, parse_doc_meta


def test_parse_meta_defaults_when_header_absent():
    meta = parse_doc_meta("# 标题\n\n正文,没有元数据头。")
    assert meta.content_type == "policy"
    assert meta.category == ""


def test_parse_meta_reads_content_type_and_category():
    md = '<!-- rag-doc: {"content_type": "faq", "category": "商品信息"} -->\n\n# 标题'
    meta = parse_doc_meta(md)
    assert meta.content_type == "faq"
    assert meta.category == "商品信息"


def test_parse_meta_bad_json_falls_back_to_defaults():
    md = '<!-- rag-doc: {"content_type": "faq", 坏 -->\n\n# 标题'
    meta = parse_doc_meta(md)
    assert meta.content_type == "policy"
    assert meta.category == ""


def test_parse_meta_unknown_content_type_falls_back_to_policy():
    md = '<!-- rag-doc: {"content_type": "video", "category": "x"} -->'
    meta = parse_doc_meta(md)
    assert meta.content_type == "policy"


def test_parse_meta_multiline_json():
    md = '<!-- rag-doc: {\n  "content_type": "manual",\n  "category": "售后"\n} -->'
    meta = parse_doc_meta(md)
    assert meta.content_type == "manual"
    assert meta.category == "售后"


def test_load_docs_discovers_md_sorted_and_parses_meta(tmp_path):
    (tmp_path / "b.md").write_text('<!-- rag-doc: {"content_type": "faq", "category": "商品信息"} -->\n# b', encoding="utf-8")
    (tmp_path / "a.md").write_text("# a\n无元数据头", encoding="utf-8")
    (tmp_path / "c.txt").write_text("not markdown", encoding="utf-8")

    docs = load_docs(tmp_path)
    assert [p.name for p, _, _ in docs] == ["a.md", "b.md"]  # 排序 + 忽略非 md
    assert docs[0][1:] == ("policy", "")  # a.md 无元数据 → 默认
    assert docs[1][1:] == ("faq", "商品信息")


def test_load_docs_single_file(tmp_path):
    f = tmp_path / "only.md"
    f.write_text('<!-- rag-doc: {"content_type": "manual", "category": "售后"} -->\n# m', encoding="utf-8")
    docs = load_docs(f)
    assert len(docs) == 1
    assert docs[0][1:] == ("manual", "售后")


def test_load_docs_missing_source_returns_empty(tmp_path):
    assert load_docs(tmp_path / "not-exist") == []
