"""chat_with_tools 服务层 TDD(spec D4 单轮架构 / Review Focus 1/2/5/6)。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.core.tools.registry import ToolRegistry
from app.models.message import Message
from app.repositories import conversation_repo, message_repo
from app.services.tool_chat import chat_with_tools


class FakeToolModel:
    """bind_tools().ainvoke() 定工具;astream() 流式收敛。second_tool_calls
    模拟 GLM 偶发二次仍要调工具的行为(触发兜底)。"""

    def __init__(
        self,
        tool_calls=None,
        reply_chunks=None,
        second_tool_calls=False,
        error=None,
        content="",
    ):
        self._tool_calls = tool_calls or []
        self._chunks = reply_chunks or [AIMessageChunk(content="好的")]
        self._second_tool_calls = second_tool_calls
        self._error = error
        self._content = content
        self.seen_messages = None

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.seen_messages = list(messages)
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content, tool_calls=self._tool_calls)

    async def astream(self, messages):
        if self._second_tool_calls:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_faq", "args": '{"keyword": "x"}', "id": "call_2", "index": 0}
                ],
            )
            return
        for chunk in self._chunks:
            yield chunk


@tool
def fake_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(fake_logistics)
    return reg


def _tool_call(name="fake_logistics", args=None, cid="call_1") -> dict:
    return {"name": name, "args": args or {"order_id": "1001"}, "id": cid, "type": "tool_call"}


async def _events(model, session, **kw):
    return [
        e
        async for e in chat_with_tools(
            "s1", "订单1001的物流到哪了", session=session, model=model,
            registry=kw.pop("registry", _registry()), **kw,
        )
    ]


async def test_no_tool_call_streams_tokens_then_done(session):
    events = await _events(FakeToolModel(), session)
    assert [e.event for e in events] == ["token", "done"]
    assert events[-1].data["session_id"] == "s1"
    assert events[-1].data["finish_reason"] == "stop"
    assert isinstance(events[-1].data["conversation_id"], int)


async def test_tool_call_emits_start_end_then_answer(session):
    model = FakeToolModel(tool_calls=[_tool_call()])
    events = await _events(model, session)
    kinds = [e.event for e in events]
    assert kinds == ["tool_start", "tool_end", "token", "done"]
    assert events[0].data["name"] == "fake_logistics"
    assert events[0].data["tool_call_id"] == "call_1"
    assert events[1].data["ok"] is True
    assert "北京转运中心" in events[1].data["summary"]


async def test_tool_trail_persisted_in_order(session):
    """Review Focus 2:user/assistant(tool_calls)/tool/assistant 四条对号落库。"""
    model = FakeToolModel(tool_calls=[_tool_call()])
    events = await _events(model, session)
    conv_id = events[-1].data["conversation_id"]

    from app.repositories import message_repo

    msgs = await message_repo.list_by_conversation(session, conv_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1].tool_calls[0]["id"] == "call_1"
    assert msgs[2].tool_call_id == "call_1"  # 对号入座
    assert "北京转运中心" in (msgs[2].content or "")


async def test_tool_failure_feeds_error_back_and_stream_continues(session):
    @tool
    def broken_tool(x: str) -> str:
        """坏工具。x: 任意"""
        raise RuntimeError("boom")

    reg = ToolRegistry()
    reg.register(broken_tool)
    model = FakeToolModel(
        tool_calls=[_tool_call(name="broken_tool", args={"x": "a"})],
    )
    events = await _events(model, session, registry=reg)
    assert events[1].event == "tool_end"
    assert events[1].data["ok"] is False
    # 流不挂起:错误已回灌,终答照常流出。
    assert events[-1].event == "done"
    assert any(e.event == "token" for e in events)


async def test_second_round_tool_calls_trigger_fallback(session):
    """spec §3:二次仍要工具 → 兜底摘要,不再执行。"""
    model = FakeToolModel(
        tool_calls=[_tool_call()],
        second_tool_calls=True,
    )
    events = await _events(model, session)
    assert events[-1].event == "done"
    fallback_tokens = [e for e in events if e.event == "token"]
    assert len(fallback_tokens) == 1
    assert "北京转运中心" in fallback_tokens[0].data["content"]


async def test_unknown_conversation_id_creates_new(session):
    """Review Focus 1:未知 conversation_id 静默新建,done 回传新 id。"""
    model = FakeToolModel()
    events = [
        e
        async for e in chat_with_tools(
            "s1", "在吗", conversation_id=99999, session=session, model=model, registry=_registry()
        )
    ]
    conv_id = events[-1].data["conversation_id"]
    assert conv_id != 99999


async def test_upstream_error_emits_error_and_persists_nothing(session):
    model = FakeToolModel(error=RuntimeError("401 unauthorized: bad key " + "x" * 500))
    events = await _events(model, session)
    assert [e.event for e in events] == ["error"]
    assert events[0].data["code"] == "UPSTREAM_ERROR"
    assert len(events[0].data["message"]) <= 200
    # CH01 语义:出错不落半截。
    from app.repositories import message_repo

    assert await message_repo.list_by_conversation(session, 999999) == []


async def test_concurrent_same_session_serialized(session):
    """同会话并发串行化(CH01 review #2 延续):第二轮的 prompt 已含第一轮历史。"""
    seen: dict = {}

    class SlowModel(FakeToolModel):
        async def astream(self, messages):
            await asyncio.sleep(0.05)
            yield AIMessageChunk(content="first-reply")

    class FastModel(FakeToolModel):
        async def astream(self, messages):
            seen["messages"] = list(messages)
            yield AIMessageChunk(content="second-reply")

    async def drain(model):
        return [e async for e in chat_with_tools(
            "s-cc", "msg-two" if model.__class__ is FastModel else "msg-one",
            session=session, model=model, registry=_registry(),
        )]

    await asyncio.gather(drain(SlowModel()), drain(FastModel()))

    from langchain_core.messages import HumanMessage

    second_human = [m.content for m in seen["messages"] if isinstance(m, HumanMessage)]
    assert second_human == ["msg-one", "msg-two"]


async def test_system_prompt_first_with_tool_guidance(session):
    model = FakeToolModel()
    await _events(model, session)
    from langchain_core.messages import SystemMessage

    assert isinstance(model.seen_messages[0], SystemMessage)
    assert "小智" in model.seen_messages[0].content
    assert "query_order" in model.seen_messages[0].content  # 工具指引在 system prompt 里


async def test_unpaired_tool_calls_degraded(session):
    """review C1:中断轮落库 [user, assistant(tool_calls)](tool 未落)时,
    下一轮回灌不得出现孤立 tool_calls(否则 GLM/OpenAI 端点 HTTP 400,
    会话被永久毒化)。预期:降级为普通 AIMessage。"""
    from app.models.message import Message

    conv = await conversation_repo.get_or_create(session, None, "s-poison")
    session.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="订单1001物流"),
            Message(
                conversation_id=conv.id,
                role="assistant",
                content=None,
                tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_x", "type": "tool_call"}],
            ),
            # 故意不落 tool 消息:模拟流式回答中用户关页面
        ]
    )
    await session.commit()

    model = FakeToolModel()
    events = [
        e
        async for e in chat_with_tools(
            "s-poison", "还在吗", session=session, model=model, registry=_registry()
        )
    ]
    assert events[-1].event == "done"
    # 回灌序列里不得有带 tool_calls 的 AIMessage。
    poisoned = [
        m for m in model.seen_messages
        if getattr(m, "tool_calls", None)
    ]
    assert poisoned == []


async def test_new_conversation_force_new(session):
    """review I2:new_conversation=True 必须真开新会话(不复用最近进行中的)。"""
    m1 = FakeToolModel()
    first = [
        e
        async for e in chat_with_tools("s-new", "第一轮", session=session, model=m1, registry=_registry())
    ]
    conv1 = first[-1].data["conversation_id"]
    m2 = FakeToolModel()
    second = [
        e
        async for e in chat_with_tools(
            "s-new", "第二轮", new_conversation=True, session=session, model=m2, registry=_registry()
        )
    ]
    assert second[-1].data["conversation_id"] != conv1


async def test_other_users_conversation_id_not_reused(session):
    """review Minor 7:显式传入他人会话 id 不得续指(静默走新建)。"""
    other = await conversation_repo.get_or_create(session, None, "someone-else")
    model = FakeToolModel()
    events = [
        e
        async for e in chat_with_tools(
            "s-own", "在吗", conversation_id=other.id, session=session, model=model, registry=_registry()
        )
    ]
    assert events[-1].data["conversation_id"] != other.id


async def test_persist_failure_emits_db_error(session):
    """review I4:落库失败必须 yield error 收尾,流不裸奔。"""
    from unittest.mock import patch

    from app.repositories import message_repo as repo_module

    model = FakeToolModel()
    with patch.object(
        repo_module, "add_message", side_effect=RuntimeError("connection dead")
    ):
        events = await _events(model, session)
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "DB_ERROR"


# ---------------------------------------------------------------- ch04 质量闸与引用


class FakeCheckModel:
    """证据自评替身:按预置 JSON 回复。"""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=self.payload)


def _cited_tool(hits: bool = True):
    """装成 query_faq:把引用写进旁路 ContextVar,返回值给模型读的编号文本。"""
    from app.core.tools.context import current_citations
    from app.rag.citations import Citation

    @tool
    def query_faq(keyword: str) -> str:
        """查知识库。keyword: 问法"""
        bucket = current_citations.get()
        if not hits:
            bucket.fill([])
            return "未在 FAQ 知识库中找到相关内容。"
        bucket.fill(
            [
                Citation(
                    n=1,
                    chunk_id=7,
                    section_path="退货退款政策 > 运费规则",
                    question="运费规则",
                    answer="满 99 元包邮,未满收 8 元。",
                    score=0.91,
                )
            ]
        )
        return "知识库命中 1 条\n[1] 章节:退货退款政策 > 运费规则\n    内容:满 99 元包邮。"

    return query_faq


def _reg_with(tool_obj) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool_obj)
    return reg


async def test_zero_hits_abstains_without_generating(session):
    """检索零命中:不进生成(不流式吐半成品),拒答并入池 retrieval_low_conf。"""
    model = FakeToolModel(tool_calls=[_tool_call(name="query_faq", args={"keyword": "货到付款"})])
    events = await _events(model, session, registry=_reg_with(_cited_tool(hits=False)))

    kinds = [e.event for e in events]
    assert kinds == ["tool_start", "tool_end", "token", "done"]
    assert events[-1].data["finish_reason"] == "abstain"
    assert events[-1].data["abstain_source"] == "retrieval_low_conf"
    assert model.seen_messages is not None  # 只有第一次定工具的调用
    assert "没有找到可靠的依据" in events[-2].data["content"]

    from sqlalchemy import select

    from app.models.low_confidence_question import LowConfidenceQuestion

    rows = (await session.execute(select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1
    assert rows[0].source == "retrieval_low_conf"
    assert rows[0].raw_question == "订单1001的物流到哪了"


async def test_weak_evidence_self_check_abstains_and_pools(session):
    """证据自评判不够:同样拒答,入池来源 self_check 并带上理由。"""
    model = FakeToolModel(tool_calls=[_tool_call(name="query_faq", args={"keyword": "会员积分"})])
    checker = FakeCheckModel('{"useful": false, "reason": "证据讲的是运费,不是积分"}')
    events = await _events(
        model, session, registry=_reg_with(_cited_tool()), check_model=checker
    )

    assert checker.calls == 1
    assert events[-1].data["finish_reason"] == "abstain"
    assert events[-1].data["abstain_source"] == "self_check"

    from sqlalchemy import select

    from app.models.low_confidence_question import LowConfidenceQuestion

    row = (await session.execute(select(LowConfidenceQuestion))).scalars().one()
    assert row.source == "self_check"
    assert "运费" in (row.reason or "")


async def test_sufficient_evidence_streams_then_emits_citations(session):
    """证据够 → 正常流式作答,并在 done 之前下发引用(编号 + 原文)。"""
    model = FakeToolModel(tool_calls=[_tool_call(name="query_faq", args={"keyword": "邮费"})])
    checker = FakeCheckModel('{"useful": true, "reason": "运费规则直接回答了包邮门槛"}')
    events = await _events(
        model, session, registry=_reg_with(_cited_tool()), check_model=checker
    )

    kinds = [e.event for e in events]
    assert kinds == ["tool_start", "tool_end", "token", "citations", "done"]
    payload = events[-2].data["citations"]
    assert payload[0]["n"] == 1
    assert payload[0]["section_path"] == "退货退款政策 > 运费规则"
    assert payload[0]["chunk_id"] == 7
    assert events[-1].data["finish_reason"] == "stop"


async def test_no_faq_call_skips_quality_gate(session):
    """没走知识库(查物流这类)不做证据自评:业务数据类的答案不该被知识闸拦。"""
    checker = FakeCheckModel('{"useful": false, "reason": "x"}')
    model = FakeToolModel(tool_calls=[_tool_call()])
    events = await _events(model, session, check_model=checker)
    assert checker.calls == 0
    assert events[-1].data["finish_reason"] == "stop"
