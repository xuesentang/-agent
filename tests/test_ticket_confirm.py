"""工单确认流单测:写操作必须经用户点确认才执行,拒了也要留痕(需求 6/7)。

三个场景:
- 点「确认提交」→ 落 tickets 表,审计记一条「成功」;
- 点「取消」→ 不落表,审计记一条「权限拒绝」;
- 恢复后模型改口(参数和用户看到的不一样)→ 按拒绝处理,不替用户提交别的东西。

注意 resume 的语义:LangGraph 的 `interrupt` 会让**整个 agent 节点从头重跑**,
所以假模型前两次都吐出同一笔调用——真实模型也该是这个行为,不该因为被挂起就改口。
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from sqlalchemy import select

from app.agent.graph import AgentDeps, build_agent_graph
from app.core.tools.context import current_conversation_id
from app.core.tools.registry import ToolMeta, ToolRegistry
from app.models.ticket import Ticket
from app.models.tool_audit import STATUS_DENIED, STATUS_OK, ToolAuditLog
from app.repositories import ticket_repo
from app.services.agent_chat import chat_with_agent

TICKET_ARGS = {"description": "退款三天没到账,请人工核实", "ticket_type": "售后"}


class IntentModel:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"intent": "物流", "confidence": 0.9}')


class CheckModel:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"useful": true, "reason": "ok"}')


class WriteModel:
    """前两次吐建单调用,第三次收尾。

    `args_seq` 让第二次(恢复时的重跑)给不同的参数,模拟真实模型改口。
    """

    def __init__(self, args: dict | None = None, args_seq: list[dict] | None = None):
        self.args = args or TICKET_ARGS
        self.args_seq = list(args_seq or [])
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls += 1
        if self.calls <= 2:
            args = self.args_seq[self.calls - 1] if len(self.args_seq) >= self.calls else self.args
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "create_ticket",
                        "args": json.dumps(args, ensure_ascii=False),
                        "id": f"call_{self.calls}",
                        "index": 0,
                    }
                ],
            )
            return
        yield AIMessageChunk(content="已为您提交工单,人工客服会尽快跟进。")


def _build(session_factory, model: WriteModel) -> tuple[AgentDeps, object]:
    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单,把问题转交人工客服处理。
        description: 问题描述; ticket_type: 工单类型,售后/投诉/咨询 三选一"""
        cid = current_conversation_id.get()
        async with session_factory() as s:
            ticket = await ticket_repo.create(s, cid, description, ticket_type)
        return f"已创建工单 {ticket.ticket_no}"

    reg = ToolRegistry()
    reg.register(create_ticket, ToolMeta(name="create_ticket", read_only=False))
    deps = AgentDeps(
        model=model,
        registry=reg,
        session_factory=session_factory,
        intent_model=IntentModel(),
        check_model=CheckModel(),
        kb_min_score=0.45,
    )
    # 同一个 graph 实例才共用 checkpointer——resume 靠它找回挂住的那一轮
    return deps, build_agent_graph(deps)


async def _run(session, deps, graph, text: str, *, conv_id=None, resume=None) -> list:
    return [
        e
        async for e in chat_with_agent(
            "s1",
            text,
            conv_id,
            resume=resume,
            session=session,
            deps=deps,
            graph=graph,
        )
    ]


async def test_approve_writes_ticket_and_audits_ok(session, session_factory):
    deps, graph = _build(session_factory, WriteModel())

    first = await _run(session, deps, graph, "退款一直没到账,帮我转人工")
    assert first[-1].event == "done"
    assert first[-1].data["finish_reason"] == "awaiting_confirm"
    assert first[-1].data["interrupted"] is True

    cards = [e for e in first if e.event == "interrupt"]
    assert len(cards) == 1
    card = cards[0].data
    assert card["type"] == "ticket_confirm"
    assert card["tool"] == "create_ticket"
    assert card["ticket_type"] == "售后"
    assert "退款" in card["description"]

    # 还没点按钮:一行都不能落
    assert list((await session.execute(select(Ticket))).scalars().all()) == []

    conv_id = first[-1].data["conversation_id"]
    second = await _run(
        session,
        deps,
        graph,
        "",
        conv_id=conv_id,
        resume={"decision": "approve", "ticket_type": card["ticket_type"], "description": card["description"]},
    )
    assert second[-1].data["finish_reason"] == "stop"
    assert any(e.event == "token" for e in second), "恢复后答复要接着流式吐出来"

    tickets = list((await session.execute(select(Ticket))).scalars().all())
    assert len(tickets) == 1
    assert tickets[0].description == TICKET_ARGS["description"]
    assert tickets[0].ticket_type == "售后"

    logs = list((await session.execute(select(ToolAuditLog))).scalars().all())
    assert [log.status for log in logs] == [STATUS_OK]
    assert logs[0].tool_name == "create_ticket"


async def test_reject_writes_nothing_and_audits_denied(session, session_factory):
    deps, graph = _build(session_factory, WriteModel())

    first = await _run(session, deps, graph, "帮我转人工")
    conv_id = first[-1].data["conversation_id"]
    card = [e for e in first if e.event == "interrupt"][0].data

    second = await _run(
        session,
        deps,
        graph,
        "",
        conv_id=conv_id,
        resume={"decision": "reject", "ticket_type": card["ticket_type"], "description": card["description"]},
    )
    # 取消不等于断线:这一轮照样要有答复收尾
    assert second[-1].event == "done"
    assert second[-1].data["finish_reason"] == "stop"

    assert list((await session.execute(select(Ticket))).scalars().all()) == []
    logs = list((await session.execute(select(ToolAuditLog))).scalars().all())
    assert [log.status for log in logs] == [STATUS_DENIED]


async def test_resume_executes_the_version_user_saw(session, session_factory):
    """恢复时节点重跑、模型改了措辞——落库的必须是用户确认过的那一版。

    否则用户点「确认」等于签了一张自己没看过的单子。
    """
    first_args = {"description": "退款三天没到账,请人工核实", "ticket_type": "售后"}
    drifted = {"description": "完全另一段话,用户没见过", "ticket_type": "售后"}
    deps, graph = _build(session_factory, WriteModel(args_seq=[first_args, drifted]))

    first = await _run(session, deps, graph, "帮我转人工")
    conv_id = first[-1].data["conversation_id"]
    card = [e for e in first if e.event == "interrupt"][0].data
    assert card["description"] == first_args["description"]

    await _run(
        session,
        deps,
        graph,
        "",
        conv_id=conv_id,
        resume={"decision": "approve", "ticket_type": card["ticket_type"], "description": card["description"]},
    )
    tickets = list((await session.execute(select(Ticket))).scalars().all())
    assert len(tickets) == 1
    assert tickets[0].description == first_args["description"], "落库内容必须是卡片上那句"


async def test_approve_without_echo_falls_back_to_model_args(session, session_factory):
    """前端没回传内容(curl 直接 approve)时退一步:用这次的参数,照样要执行。"""
    deps, graph = _build(session_factory, WriteModel())
    first = await _run(session, deps, graph, "帮我转人工")
    conv_id = first[-1].data["conversation_id"]

    await _run(session, deps, graph, "", conv_id=conv_id, resume="approve")
    tickets = list((await session.execute(select(Ticket))).scalars().all())
    assert len(tickets) == 1
    assert tickets[0].description == TICKET_ARGS["description"]


async def test_read_only_turn_has_no_interrupt(session, session_factory):
    """只读工具不该弹卡片:确认闸只认写操作。"""

    @tool
    def query_order(order_id: str) -> str:
        """查订单。order_id: 订单号"""
        return f"订单 {order_id} 已发货"

    class ReadModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "query_order", "args": '{"order_id": "1001"}', "id": "c1", "index": 0}
                    ],
                )
                return
            yield AIMessageChunk(content="订单 1001 已发货。")

    reg = ToolRegistry()
    reg.register(query_order)
    deps = AgentDeps(
        model=ReadModel(),
        registry=reg,
        session_factory=session_factory,
        intent_model=IntentModel(),
        check_model=CheckModel(),
    )
    events = await _run(session, deps, build_agent_graph(deps), "订单 1001 到哪了")
    assert not [e for e in events if e.event == "interrupt"]
    assert events[-1].data["finish_reason"] == "stop"
    assert [e.data["ok"] for e in events if e.event == "tool_end"] == [True]
