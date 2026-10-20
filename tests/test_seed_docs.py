"""seed_docs 单测:种子文档覆盖切分引擎全部分支 + 落盘行为。

分支断言直接对应提示词第三章「文档处理」四条要求与
indexer 三格映射的 faq/policy 两条分支。
"""

import pytest

from app.rag.chunker import ChunkConfig, split_document
from app.rag.doc_loader import parse_doc_meta
from app.rag.indexer import to_knowledge_chunks
from app.rag.seed_docs import SEED_DOCS, write_seed_docs


@pytest.fixture(scope="module")
def policy_raws():
    return split_document(SEED_DOCS["return_policy.md"])


@pytest.fixture(scope="module")
def faq_raws():
    return split_document(SEED_DOCS["product_faq.md"])


@pytest.fixture(scope="module")
def manual_raws():
    return split_document(SEED_DOCS["after_sales_manual.md"])


# ---------------------------------------------------------------- 文档头元数据


def test_every_seed_doc_has_valid_meta():
    for name, md in SEED_DOCS.items():
        meta = parse_doc_meta(md)
        assert meta.content_type in ("faq", "policy", "manual"), name
        assert meta.category, name  # 每份文档都带文档级分类


def test_every_seed_doc_declared_ai_synthetic():
    for name, md in SEED_DOCS.items():
        assert "AI 合成数据" in md.splitlines()[0], name


def test_seed_docs_cover_three_content_types():
    metas = {name: parse_doc_meta(md).content_type for name, md in SEED_DOCS.items()}
    assert set(metas.values()) == {"policy", "faq", "manual"}


# ---------------------------------------------------------------- policy 分支


def test_policy_multilevel_heading_path(policy_raws):
    """多级标题:至少存在三级路径(章 > 节 > 小节)。"""
    paths = {rc.section_path for rc in policy_raws}
    assert any(path.count(" > ") >= 2 for path in paths)


def test_policy_long_section_split_into_multiple_chunks(policy_raws):
    """超长章节递归切:退款时效一节被切成多块,且相邻块有重叠。"""
    heads = [rc for rc in policy_raws if rc.heading == "退款时效"]
    assert len(heads) >= 2
    # 相邻块重叠:后块开头是前块结尾的完整句子序列
    assert heads[1].body[:20] in heads[0].body[-80:] or heads[0].body[-40:] in heads[1].body


def test_policy_no_half_sentence_at_chunk_edges(policy_raws):
    """块尾不得半截话:每个文本块以句末符结尾(允许冒号列表尾巴)。"""
    for rc in policy_raws:
        if rc.is_table:
            continue
        assert rc.body.rstrip()[-1:] in "。！？!?;；:", rc.section_path


def test_policy_key_clause_flagged(policy_raws):
    """关键条款标记:含「必须/禁止」的节被标 is_key_clause。"""
    flagged = {rc.heading for rc in policy_raws if rc.is_key_clause}
    assert "商品完好标准" in flagged or "不支持退货的情形" in flagged


def test_policy_three_field_mapping(policy_raws):
    """三格映射(policy):questions=末级标题,category=上级标题路径,元数据不进向量文本。"""
    chunks = to_knowledge_chunks(policy_raws, "policy")
    target = next(c for c in chunks if c.questions == "运费规则")
    assert target.category == "退货退款政策"
    assert "邮费" not in target.embedding_text()  # 靶心词只在 query 侧出现
    assert "运费" in target.embedding_text()
    assert target.content_type == "policy"
    assert target.section_path == "退货退款政策 > 运费规则"


# ---------------------------------------------------------------- faq 分支


def test_faq_questions_are_real_user_phrasings(faq_raws):
    """faq 分支:questions 填真实问法(Q: 行),不是章节标题。"""
    chunks = to_knowledge_chunks(faq_raws, "faq")
    questions = {q for c in chunks for q in c.questions.splitlines()}
    assert "跑鞋偏码吗?" in questions
    assert "电器保修多久?" in questions
    assert not any(q.startswith("##") or q in ("尺码与版型", "发票") for q in questions)


def test_faq_doc_category_used_for_all_chunks(faq_raws):
    """faq 分支:category 用文档级分类(元数据头),不用标题路径。"""
    chunks = to_knowledge_chunks(faq_raws, "faq", doc_category="商品信息")
    assert {c.category for c in chunks} == {"商品信息"}


# ---------------------------------------------------------------- manual 分支


def test_manual_big_table_split_with_header_copied(manual_raws):
    """大表格按行切,每块都复制表头(表头行 + 分隔行)。"""
    tables = [rc for rc in manual_raws if rc.is_table]
    assert len(tables) >= 2, "7 行时效表应被切成多块"
    for rc in tables:
        assert rc.body.startswith("| 工单类型 |"), rc.body[:40]
        assert "|---|---|" in rc.body.splitlines()[1]


def test_manual_table_chunks_within_max_chars(manual_raws):
    """表格块不超 max_chars(400)。"""
    for rc in manual_raws:
        if rc.is_table:
            assert len(rc.body) <= ChunkConfig().max_chars


# ---------------------------------------------------------------- 落盘行为


def test_write_seed_docs_creates_all_files(tmp_path):
    written, skipped = write_seed_docs(tmp_path)
    assert sorted(p.name for p in written) == sorted(SEED_DOCS)
    assert skipped == []
    assert (tmp_path / "return_policy.md").exists()


def test_write_seed_docs_skips_existing_without_force(tmp_path):
    write_seed_docs(tmp_path)
    written, skipped = write_seed_docs(tmp_path)
    assert written == []
    assert len(skipped) == len(SEED_DOCS)


def test_write_seed_docs_force_overwrites(tmp_path):
    write_seed_docs(tmp_path)
    (tmp_path / "return_policy.md").write_text("被改过的内容", encoding="utf-8")
    written, _ = write_seed_docs(tmp_path, force=True)
    assert len(written) == len(SEED_DOCS)
    assert "退货退款政策" in (tmp_path / "return_policy.md").read_text(encoding="utf-8")
