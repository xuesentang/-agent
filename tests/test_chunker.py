"""chunker 单测(ch03 TDD 主战场)。

覆盖提示词第三章文档处理要求:标题层级结构感知、超长递归切、
块间重叠裁到最近句号(不留半截话)、大表格按行切复制表头。
"""

from app.rag.chunker import ChunkConfig, RawChunk, split_document

SENT_ENDINGS = ("。", "！", "？", "!", "?", "；", ";")


def assert_no_half_sentence(chunk: RawChunk):
    """块尾必须是完整句末(表格块除外);块首不得以小写字母/逗号开头(断句残留)。"""
    if chunk.is_table:
        return
    body = chunk.body.rstrip()
    assert body, f"空块:{chunk.section_path}"
    assert body[-1] in SENT_ENDINGS, f"块尾半截话:{body[-30:]}"


# ---------------------------------------------------------------- 基础结构


def test_single_short_section_one_chunk():
    md = """# 退货政策

签收后 7 天内可申请无理由退货,商品需保持吊牌齐全。"""
    chunks = split_document(md)
    assert len(chunks) == 1
    assert chunks[0].section_path == "退货政策"
    assert "7 天内" in chunks[0].body
    assert_no_half_sentence(chunks[0])


def test_nested_headings_build_path():
    md = """# 售后手册

## 退货流程

### 申请阶段

用户在订单页点击申请退货。"""
    chunks = split_document(md)
    assert len(chunks) == 1
    assert chunks[0].section_path == "售后手册 > 退货流程 > 申请阶段"


def test_empty_and_whitespace_input():
    assert split_document("") == []
    assert split_document("   \n\n  ") == []


def test_no_heading_document_is_single_implicit_section():
    md = "本店支持 7 天无理由退货。运费满 99 元包邮。"
    chunks = split_document(md)
    assert len(chunks) == 1
    assert chunks[0].section_path == ""
    assert_no_half_sentence(chunks[0])


def test_multiple_sibling_sections_keep_order():
    md = """# 政策

## 退货

支持 7 天无理由退货。

## 换货

支持 15 天免费换货。"""
    chunks = split_document(md)
    assert [c.section_path for c in chunks] == ["政策 > 退货", "政策 > 换货"]


# ---------------------------------------------------------------- 递归切分


def test_long_section_splits_into_bounded_chunks():
    para = "这是一段关于退货流程的详细说明文字,每个句子都很完整。"
    body = "".join(para for _ in range(30))  # ~600 字,超 max_chars=400
    md = f"# 退货政策\n\n{body}"
    chunks = split_document(md)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.body) <= 400, f"超限:{len(c.body)}"
        assert_no_half_sentence(c)


def test_overlap_shares_sentence_between_neighbors():
    para = "这是一段关于退货流程的详细说明文字,每个句子都很完整。"
    body = "".join(para for _ in range(30))
    chunks = split_document(md := f"# 退货政策\n\n{body}")
    assert len(chunks) >= 2
    # 相邻块必须有共享内容(重叠),且共享部分是完整句子
    overlap = _common_suffix_prefix(chunks[0].body, chunks[1].body)
    assert overlap, "相邻块没有重叠"
    assert overlap.rstrip()[-1] in SENT_ENDINGS, f"重叠区半截话:{overlap[-30:]}"


def _common_suffix_prefix(a: str, b: str) -> str:
    """a 的后缀与 b 的前缀的最长公共部分(句子粒度粗查:找 b 的首句在 a 中的位置)。"""
    first_sentence_b = _first_sentence(b)
    idx = a.find(first_sentence_b)
    return a[idx:] if idx >= 0 else ""


def _first_sentence(text: str) -> str:
    for i, ch in enumerate(text):
        if ch in SENT_ENDINGS:
            return text[: i + 1]
    return text


# ---------------------------------------------------------------- 表格


def test_long_table_split_by_rows_with_header_copied():
    header = "| 商品 | 退货条件 |\n| --- | --- |"
    rows = "\n".join(f"| 商品{i} | 条件{i} |" for i in range(20))
    md = f"# 退货政策\n\n{header}\n{rows}"
    chunks = split_document(md, ChunkConfig(max_chars=200))
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) >= 2, "长表格没按行切"
    for c in table_chunks:
        assert "| 商品 |" in c.body and "| --- |" in c.body, "表格块缺表头"
    # 数据行不丢不重(排除每块复制的表头行本身)
    header_line = "| 商品 | 退货条件 |"
    data_rows_out = [
        ln
        for c in table_chunks
        for ln in c.body.splitlines()
        if ln.startswith("| 商品") and ln != header_line
    ]
    assert len(data_rows_out) == 20
    assert len(set(data_rows_out)) == 20


def test_short_table_stays_single_chunk():
    md = """# 政策

| 商品 | 条件 |
| --- | --- |
| T恤 | 7 天 |
| 跑鞋 | 7 天 |"""
    chunks = split_document(md)
    assert len(chunks) == 1
    assert chunks[0].is_table
    assert_no_half_sentence(chunks[0])


# ---------------------------------------------------------------- 关键条款


def test_key_clause_detection():
    md = """# 退货政策

## 重要提示

退货时必须保持商品吊牌齐全,否则影响退款。"""
    chunks = split_document(md)
    assert chunks[0].is_key_clause is True


def test_non_key_clause_default_false():
    md = "# 退货政策\n\n一般情况下支持退货。"
    assert split_document(md)[0].is_key_clause is False


# ---------------------------------------------------------------- 配置


def test_config_max_chars_respected():
    body = "".join("这是完整的一句话,说明退货政策。" for _ in range(40))
    chunks = split_document(f"# 政策\n\n{body}", ChunkConfig(max_chars=120))
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.body) <= 120


# ---------------------------------------------------------------- HTML 注释


def test_html_comment_header_stripped_from_content():
    """文档头的 HTML 元数据注释不是知识内容,不进任何 chunk。"""
    md = """<!-- rag-doc: {"content_type": "policy"} -->

# 退货政策

正文说明退货规则。"""
    chunks = split_document(md)
    assert len(chunks) == 1
    assert "rag-doc" not in chunks[0].body
    assert chunks[0].body == "正文说明退货规则。"


def test_multiline_html_comment_stripped():
    md = """<!-- 多行注释
{"content_type": "policy",
 "category": "退换货"}
-->

# 标题

正文。"""
    chunks = split_document(md)
    assert len(chunks) == 1
    assert "content_type" not in chunks[0].body
