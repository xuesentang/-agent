"""上下文接线的单测:两行 ctx 日志、State 消息累积、降级推进锚点、超预算起摘要。"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.agent import graph as graph_module
from app.agent.graph import AgentDeps
from app.core.token_budget import ContextBudget
from app.core.tools.registry import ToolRegistry
from app.repositories import conversation_repo, conversation_summary_repo, message_repo
from app.services import agent_chat
from app.services.agent_chat import chat_with_agent, _plan_context


class IntentModel:
    def __init__(self, intent: str = "物流"):
        self.intent = intent

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"intent": "%s", "confidence": 0.9}' % self.intent)


class CheckModel:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"useful": true, "reason": "r"}')


class StreamAgentModel:
    def __init__(self, texts=("好的,", "已处理。")):
        self.texts = list(texts)
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.seen.append(list(messages))
        for t in self.texts:
            yield AIMessageChunk(content=t)


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return "已发货"


def _deps(intent: str = "物流") -> AgentDeps:
    reg = ToolRegistry()
    reg.register(fake_order)
    return AgentDeps(
        model=StreamAgentModel(),
        registry=reg,
        session_factory=None,
        intent_model=IntentModel(intent),
        check_model=CheckModel(),
        kb_min_score=0.45,
    )


async def _run(session, deps, text: str, **kw):
    return [e async for e in chat_with_agent("s-ctx", text, session=session, deps=deps, **kw)]


def _budget(layer1: int, layer2: int) -> ContextBudget:
    return ContextBudget(
        window=18000,
        reserved_system=0,
        reserved_evidence=0,
        reserved_summary=0,
        reserved_output=0,
        reserved_user_input=0,
        reserved_react_peak=0,
        safety=0,
        history_total=layer1 + layer2,
        layer1=layer1,
        layer2=layer2,
    )


async def test_history_ctx_logged_every_turn_including_chitchat(session, monkeypatch, caplog):
    monkeypatch.setattr(agent_chat, "budget_from_settings", lambda _p: _budget(10_000, 10_000))
    with caplog.at_level(logging.INFO, logger="app.services.agent_chat"):
        await _run(session, _deps("闲聊"), "在吗")
    assert "history_ctx" in caplog.text


async def test_state_messages_accumulate_via_reducer(session, monkeypatch):
    """需求 5:State 的 messages 挂 add_messages,节点只吐新消息,跨轮累积。"""
    monkeypatch.setattr(agent_chat, "budget_from_settings", lambda _p: _budget(10_000, 10_000))
    deps = _deps("物流")
    first = await _run(session, deps, "订单 1001 到哪了")
    conv_id = first[-1].data["conversation_id"]
    await _run(session, deps, "那 1002 呢", conversation_id=conv_id)

    from app.agent.graph import build_agent_graph

    graph = build_agent_graph(deps)
    snapshot = await graph.aget_state({"configurable": {"thread_id": str(conv_id)}})
    messages = snapshot.values.get("messages") or []
    roles = [m.type for m in messages]
    assert roles.count("human") == 2, f"两轮各一条用户消息,实得 {roles}"
    assert roles.count("ai") == 2, f"两轮各一条答复,实得 {roles}"


async def test_layer1_over_budget_demotes_and_moves_anchor(session, monkeypatch):
    monkeypatch.setattr(agent_chat, "budget_from_settings", lambda _p: _budget(20, 10_000))
    conv = await conversation_repo.get_or_create(session, None, "u-demote")
    for i in range(6):
        await message_repo.add_message(
            session, conv.id, role="user" if i % 2 == 0 else "assistant", content=f"第{i}条" + "内容" * 30
        )
    assert conv.layer1_from_msg_id == 0
    plan, summary, _ = await _plan_context(session, conv, "当前这句")
    assert plan.demoted > 0
    assert plan.layer1_from_msg_id > 0
    await session.refresh(conv)
    assert conv.layer1_from_msg_id == plan.layer1_from_msg_id, "降级要把锚点推到新的层 1 起点"
    # 预算再小也留最后一条:层 1 空掉等于把用户刚说的话丢了
    assert len(plan.layer1) == 1
    assert plan.layer1[-1].content.startswith("第5条")


async def test_summary_is_scheduled_when_layer2_over_budget(session, monkeypatch):
    """层 2 超预算 → 真的起后台摘要(不阻塞本轮)。"""
    from app.services import summary_service

    calls: list[dict] = []

    async def fake_run_summary(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "seq": 1}

    monkeypatch.setattr(summary_service, "run_summary", fake_run_summary)
    monkeypatch.setattr(agent_chat, "budget_from_settings", lambda _p: _budget(20, 5))
    conv = await conversation_repo.get_or_create(session, None, "s-ctx")  # 用户键要与 _run 的 session_id 一致
    for i in range(6):
        await message_repo.add_message(
            session, conv.id, role="user" if i % 2 == 0 else "assistant", content=f"第{i}条" + "内容" * 30
        )
    await _run(session, _deps("物流"), "还要等多久", conversation_id=conv.id)
    await asyncio.sleep(0)  # 让后台任务跑完(生产代码不 await)
    assert calls, "层 2 超预算必须起后台摘要"
    assert calls[0]["layer1_from_msg_id"] > 0


async def test_summary_not_scheduled_within_budget(session, monkeypatch):
    from app.services import summary_service

    calls: list[dict] = []

    async def fake_run_summary(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(summary_service, "run_summary", fake_run_summary)
    monkeypatch.setattr(agent_chat, "budget_from_settings", lambda _p: _budget(10_000, 10_000))
    conv = await conversation_repo.get_or_create(session, None, "s-ctx")
    await message_repo.add_message(session, conv.id, role="user", content="问题" * 50)
    await _run(session, _deps("物流"), "再来一句", conversation_id=conv.id)
    await asyncio.sleep(0)
    assert calls == [], "装得下就不压(验收 3)"
