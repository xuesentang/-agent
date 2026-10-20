"""miner 单测:抽取落暂存、去重三处比对、入库 pending、dry_run、容错(SQLite + FakeExtractor)。"""

import pytest

from app.models.knowledge_chunk import KnowledgeChunk
from app.rag.miner import QaPair, MineReport, list_minable_conversation_ids, mine_qa_pairs
from app.repositories import conversation_repo, knowledge_repo, message_repo, qa_staging_repo


async def _make_conversation(session_factory, user_id: str, turns: list[tuple[str, str]]) -> int:
    """造一通会话:user/assistant 交替落库,返回 conversation_id。"""
    async with session_factory() as session:
        conv = await conversation_repo.get_or_create(session, None, user_id)
        for role, content in turns:
            await message_repo.add_message(session, conv.id, role, content)
        return conv.id


class FakeExtractor:
    """按会话 id 返回预定 QA 对;exc_for 里的会话抛错(测容错)。"""

    def __init__(self, plan: dict[int, list[QaPair]], exc_for: set[int] | None = None):
        self.plan = plan
        self.exc_for = exc_for or set()
        self.calls: list[int] = []

    async def __call__(self, messages):
        conv_id = messages[0].conversation_id
        self.calls.append(conv_id)
        if conv_id in self.exc_for:
            raise RuntimeError("llm exploded")
        return self.plan.get(conv_id, [])


# ---------------------------------------------------------------- 基础流程


async def test_mine_extracts_stages_and_keeps_new(session_factory):
    conv_id = await _make_conversation(
        session_factory,
        "u1",
        [("user", "运费怎么算"), ("assistant", "满 99 包邮,否则 8 元。")],
    )
    extractor = FakeExtractor({conv_id: [QaPair(question="运费怎么算", answer="满 99 包邮,否则 8 元。")]})

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)

    assert report.batches == 1 and report.conversations == 1
    assert report.extracted == 1 and report.kept == 1 and report.discarded == 0

    async with session_factory() as session:
        kept_rows = await qa_staging_repo.list_by_status(session, "kept")
        extracted_rows = await qa_staging_repo.list_by_status(session, "extracted")
        chunks = await knowledge_repo.get_by_ids(
            session, [c.id for c in (await session.execute(
                __import__("sqlalchemy").select(KnowledgeChunk)
            )).scalars()]
        )
    assert len(kept_rows) == 1 and extracted_rows == []
    assert kept_rows[0].question == "运费怎么算"
    assert len(chunks) == 1
    assert chunks[0].content_type == "qa_mined"
    assert chunks[0].category == "对话挖掘"
    assert chunks[0].vectorize_status == "pending"  # 等 build_knowledge 补向量化


async def test_mine_rerun_second_time_all_discarded(session_factory):
    conv_id = await _make_conversation(
        session_factory, "u1", [("user", "保修多久"), ("assistant", "12 个月。")]
    )
    extractor = FakeExtractor({conv_id: [QaPair(question="保修多久", answer="12 个月。")]})

    first = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)
    assert first.kept == 1

    second = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)
    assert second.extracted == 1 and second.kept == 0 and second.discarded == 1

    async with session_factory() as session:
        from sqlalchemy import func, select

        count = (
            await session.execute(select(func.count()).select_from(KnowledgeChunk))
        ).scalar()
    assert count == 1  # 重复执行不入重复知识


async def test_mine_dedupes_against_existing_knowledge_chunks(session_factory):
    conv_id = await _make_conversation(
        session_factory, "u1", [("user", "能开发票吗"), ("assistant", "可以。")]
    )
    async with session_factory() as session:
        await knowledge_repo.add_chunks(
            session,
            [KnowledgeChunk(category="发票", questions="能开发票吗", answer="支持电子发票。")],
        )
    extractor = FakeExtractor({conv_id: [QaPair(question="能开发票吗", answer="可以,下单时填写抬头。")]})

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)
    assert report.kept == 0 and report.discarded == 1  # 问法已存在 → 判重


async def test_mine_dedupes_within_batch(session_factory):
    """同一问法在两个会话都被抽出:第一个 kept,第二个 discarded(seen 跨批累积)。"""
    conv1 = await _make_conversation(session_factory, "u1", [("user", "怎么退货"), ("assistant", "7 天内。")])
    conv2 = await _make_conversation(session_factory, "u2", [("user", "怎么退货"), ("assistant", "同样 7 天。")])
    extractor = FakeExtractor(
        {
            conv1: [QaPair(question="怎么退货", answer="签收 7 天内可退。")],
            conv2: [QaPair(question="怎么退货?", answer="吊牌完好即可。")],  # 规范化后同问法
        }
    )

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor, batch_size=20)
    assert report.kept == 1 and report.discarded == 1


# ---------------------------------------------------------------- 容错与开关


async def test_mine_sanitizes_dirty_extractor_output(session_factory):
    """extractor 输出脏数据在 miner 边界被净化:空问法/空答案/问答相同都不进流水线。"""
    conv_id = await _make_conversation(
        session_factory, "u1", [("user", "在吗"), ("assistant", "在的。")]
    )
    extractor = FakeExtractor(
        {
            conv_id: [
                QaPair(question="  ", answer="有答案没问题"),  # 空问法
                QaPair(question="正常问题", answer=""),  # 空答案
                QaPair(question="同一句话", answer="同一句话"),  # 问答相同
                QaPair(question="正常问题", answer="有效答案。"),  # 唯一干净的一条
            ]
        }
    )

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)
    assert report.extracted == 1 and report.kept == 1 and report.discarded == 0


async def test_mine_survives_extractor_failure_for_one_conversation(session_factory):
    conv1 = await _make_conversation(session_factory, "u1", [("user", "问题一"), ("assistant", "答一。")])
    conv2 = await _make_conversation(session_factory, "u2", [("user", "问题二"), ("assistant", "答二。")])
    extractor = FakeExtractor(
        {
            conv1: [QaPair(question="问题一", answer="答一。")],
            conv2: [QaPair(question="问题二", answer="答二。")],
        },
        exc_for={conv2},
    )

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor)
    assert report.kept == 1  # conv2 抽坏不影响 conv1


async def test_mine_dry_run_writes_nothing(session_factory):
    conv_id = await _make_conversation(
        session_factory, "u1", [("user", "有现货吗"), ("assistant", "有。")]
    )
    extractor = FakeExtractor({conv_id: [QaPair(question="有现货吗", answer="详情页显示实时库存。")]})

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor, dry_run=True)
    assert report.kept == 1

    async with session_factory() as session:
        from sqlalchemy import func, select

        staging = (await session.execute(select(func.count()).select_from(__import__("app.models.qa_extraction_staging", fromlist=["QaExtractionStaging"]).QaExtractionStaging))).scalar()
        chunks = (await session.execute(select(func.count()).select_from(KnowledgeChunk))).scalar()
    assert staging == 0 and chunks == 0


async def test_mine_batch_write_is_atomic(session_factory, monkeypatch):
    """批次中途写失败:暂存与知识库都不留半截。

    半截的后果不只是脏数据——kept 的问法重跑时已被 seen 判重,永远补不回知识库。
    """
    conv_id = await _make_conversation(
        session_factory, "u1", [("user", "支持花呗吗"), ("assistant", "支持,下单时可选。")]
    )
    extractor = FakeExtractor({conv_id: [QaPair(question="支持花呗吗", answer="支持,下单时可选。")]})

    async def boom(*args, **kwargs):
        raise RuntimeError("knowledge insert failed")

    monkeypatch.setattr(knowledge_repo, "add_chunks", boom)
    with pytest.raises(RuntimeError):
        await mine_qa_pairs(session_factory=session_factory, extractor=extractor)

    from sqlalchemy import func, select

    from app.models.qa_extraction_staging import QaExtractionStaging

    async with session_factory() as session:
        staging = (
            await session.execute(select(func.count()).select_from(QaExtractionStaging))
        ).scalar()
        chunks = (await session.execute(select(func.count()).select_from(KnowledgeChunk))).scalar()
    assert staging == 0 and chunks == 0


async def test_mine_empty_db_returns_zero_report(session_factory):
    report = await mine_qa_pairs(session_factory=session_factory, extractor=FakeExtractor({}))
    assert report == MineReport()


async def test_mine_batch_size_controls_batches(session_factory):
    for i in range(3):
        await _make_conversation(session_factory, f"u{i}", [("user", f"问题{i}"), ("assistant", f"答{i}。")])
    extractor = FakeExtractor({})

    report = await mine_qa_pairs(session_factory=session_factory, extractor=extractor, batch_size=2)
    assert report.batches == 2 and report.conversations == 3  # 2 + 1


async def test_mine_ignores_conversations_without_user_messages(session_factory):
    async with session_factory() as session:
        conv = await conversation_repo.get_or_create(session, None, "u1")
        await message_repo.add_message(session, conv.id, "assistant", "只有客服在说话。")
    async with session_factory() as session:
        assert await list_minable_conversation_ids(session) == []


async def test_max_batches_limits_scan(session_factory):
    for i in range(4):
        await _make_conversation(session_factory, f"u{i}", [("user", f"问题{i}"), ("assistant", f"答{i}。")])
    extractor = FakeExtractor({})

    report = await mine_qa_pairs(
        session_factory=session_factory, extractor=extractor, batch_size=1, max_batches=2
    )
    assert report.batches == 2 and report.conversations == 2
