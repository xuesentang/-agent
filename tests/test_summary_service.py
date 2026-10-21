"""后台摘要单测:触发判定、落库推锚点、失败不影响主链路、非阻塞。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from app.repositories import conversation_repo, conversation_summary_repo, message_repo
from app.services.summary_service import (
    run_summary,
    schedule_summary,
    should_summarize,
)


class FakeSummaryModel:
    def __init__(self, content: str = "用户问过 W40,报过订单 1001,要求退货。", error: Exception | None = None):
        self.content = content
        self.error = error
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kwargs):
        self.prompts.append(messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1]))
        if self.error:
            raise self.error
        return AIMessage(content=self.content)


async def _seed(session, conv_id: int, n: int) -> None:
    for i in range(n):
        await message_repo.add_message(
            session, conv_id, role="user" if i % 2 == 0 else "assistant", content=f"第{i}条消息" + "内容" * 20
        )


def test_trigger_uses_usage_not_count():
    assert should_summarize(2000, 1695) is True
    assert should_summarize(1000, 1695) is False
    assert should_summarize(1000, 0) is False  # 预算为 0 说明整体装不下,不压


async def test_run_summary_writes_segment_and_moves_anchors(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u1")
    await _seed(session, conv.id, 6)
    msgs = await message_repo.list_by_conversation(session, conv.id)

    out = await run_summary(
        conversation_id=conv.id,
        session_factory=session_factory,
        model=FakeSummaryModel(),
        layer1_from_msg_id=msgs[3].id,
    )
    assert out["ok"] is True and out["seq"] == 1

    rows = await conversation_summary_repo.list_by_conversation(session, conv.id)
    assert len(rows) == 1
    assert rows[0].from_msg_id == msgs[0].id
    assert rows[0].to_msg_id == msgs[2].id  # 只压到层 1 起点之前

    await session.refresh(conv)
    assert conv.summary_upto_msg_id == msgs[2].id
    assert conv.layer1_from_msg_id == msgs[3].id  # 边界追到层 1 起点


async def test_second_summary_appends_a_new_segment(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u2")
    await _seed(session, conv.id, 4)
    msgs = await message_repo.list_by_conversation(session, conv.id)
    await run_summary(conversation_id=conv.id, session_factory=session_factory, model=FakeSummaryModel("第一段"), layer1_from_msg_id=msgs[2].id)

    await _seed(session, conv.id, 2)
    refreshed = await message_repo.list_by_conversation(session, conv.id)
    out = await run_summary(
        conversation_id=conv.id, session_factory=session_factory, model=FakeSummaryModel("第二段"), layer1_from_msg_id=refreshed[-1].id
    )
    assert out["seq"] == 2
    rows = await conversation_summary_repo.list_by_conversation(session, conv.id)
    assert [r.content for r in rows] == ["第一段", "第二段"]
    # 第二段从第一段之后开始,不重叠
    assert rows[1].from_msg_id > rows[0].to_msg_id


async def test_model_failure_returns_not_ok_and_writes_nothing(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u3")
    await _seed(session, conv.id, 4)
    msgs = await message_repo.list_by_conversation(session, conv.id)
    out = await run_summary(
        conversation_id=conv.id,
        session_factory=session_factory,
        model=FakeSummaryModel(error=RuntimeError("upstream 500")),
        layer1_from_msg_id=msgs[-1].id,
    )
    assert out["ok"] is False and out["reason"] == "model_error"
    assert await conversation_summary_repo.list_by_conversation(session, conv.id) == []
    await session.refresh(conv)
    assert conv.summary_upto_msg_id == 0


async def test_empty_range_is_skipped(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u4")
    out = await run_summary(conversation_id=conv.id, session_factory=session_factory, model=FakeSummaryModel())
    assert out["ok"] is False and out["reason"] == "empty_range"


async def test_schedule_returns_none_when_within_budget(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u5")
    task = schedule_summary(
        conversation_id=conv.id, session_factory=session_factory, layer2_tokens=100, layer2_budget=1695, layer1_from_msg_id=1
    )
    assert task is None


async def test_schedule_runs_in_background_task(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u6")
    await _seed(session, conv.id, 4)
    msgs = await message_repo.list_by_conversation(session, conv.id)
    task = schedule_summary(
        conversation_id=conv.id,
        session_factory=session_factory,
        layer2_tokens=5000,
        layer2_budget=1695,
        layer1_from_msg_id=msgs[-1].id,
        model=FakeSummaryModel(),
    )
    assert task is not None, "超预算必须起后台任务"
    out = await task  # 单测等它,生产代码不等
    assert out["ok"] is True


async def test_second_trigger_is_skipped_while_first_is_running(session, session_factory, monkeypatch):
    """压一段要一两秒,这期间用户可能又发一句:不能叠着压两遍同一批。"""
    from app.services import summary_service

    conv = await conversation_repo.get_or_create(session, None, "u8")
    await _seed(session, conv.id, 4)
    msgs = await message_repo.list_by_conversation(session, conv.id)
    gate = asyncio.Event()

    async def slow_run_summary(**_kwargs):
        await gate.wait()
        return {"ok": True, "seq": 1}

    monkeypatch.setattr(summary_service, "run_summary", slow_run_summary)
    kwargs = dict(
        conversation_id=conv.id,
        session_factory=session_factory,
        layer2_tokens=5000,
        layer2_budget=1695,
        layer1_from_msg_id=msgs[-1].id,
    )
    first = schedule_summary(**kwargs)
    second = schedule_summary(**kwargs)
    assert first is not None
    assert second is None, "上一段还在跑时不该再起一个"
    gate.set()
    await first
    # 跑完之后必须能再起:去重表要清干净,否则摘要从此再也不触发
    third = schedule_summary(**kwargs)
    assert third is not None, "上一个跑完就该放行"
    gate.set()
    await third


async def test_summary_prompt_forbids_invention_and_chitchat(session, session_factory):
    conv = await conversation_repo.get_or_create(session, None, "u7")
    await _seed(session, conv.id, 2)
    msgs = await message_repo.list_by_conversation(session, conv.id)
    model = FakeSummaryModel()
    await run_summary(conversation_id=conv.id, session_factory=session_factory, model=model, layer1_from_msg_id=msgs[-1].id)
    prompt = model.prompts[0]
    assert "一个字都不许加" in prompt
    assert "寒暄" in prompt
    assert "道具" not in prompt  # 防串行:模板没被别处的字符串顶掉
