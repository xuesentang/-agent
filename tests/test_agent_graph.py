"""图骨架单测:四个出口各走一遍 + 知识类必过检索与闸 + 同 thread 状态留存。

检索那一步用替身(真检索已由 CH04 的单测与 eval 覆盖),这里要验的是**接线**:
知识类必须走到 retrieve_knowledge,业务类必须绕开它。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.agent import graph as graph_module
from app.agent.graph import (
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    AgentDeps,
    build_agent_graph,
)
from app.core.tools.registry import ToolRegistry
from app.rag.selfcheck import ABSTAIN_REPLY


class ScriptedIntentModel:
    def __init__(self, intent: str = "售后"):
        self.intent = intent
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=f'{{"intent": "{self.intent}", "confidence": 0.9}}')


class ScriptedAgentModel:
    """一步收敛的 Agent:只吐一段正文。"""

    def __init__(self, text: str = "好的,已为您处理。", calls_ref: list | None = None):
        self.text = text
        self.calls = 0
        self.ref = calls_ref

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls += 1
        if self.ref is not None:
            self.ref.append(self.calls)
        yield AIMessageChunk(content=self.text)


class ScriptedCheckModel:
    def __init__(self, useful: bool = True, reason: str = "ok"):
        self.useful = useful
        self.reason = reason
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content='{"useful": %s, "reason": "%s"}' % (str(self.useful).lower(), self.reason))


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return f"订单 {order_id} 已发货"


def _hits(scores: list[float]):
    return [
        SimpleNamespace(
            chunk=SimpleNamespace(
                id=i + 1,
                section_path="退货退款政策 > 运费规则",
                questions="退货运费谁承担",
                answer="七日无理由退货的寄回运费由消费者承担。",
            ),
            score=s,
        )
        for i, s in enumerate(scores)
    ]


def _deps(intent="售后", useful=True, checkpointer=None, **kw):
    reg = ToolRegistry()
    reg.register(fake_order)
    model = ScriptedAgentModel()
    check = ScriptedCheckModel(useful=useful)
    deps = AgentDeps(
        model=model,
        registry=reg,
        session_factory=None,
        intent_model=ScriptedIntentModel(intent),
        check_model=check,
        kb_min_score=0.45,
        **({"checkpointer": checkpointer} if checkpointer else {}),
        **kw,
    )
    return deps, model, check


@pytest.fixture
def patch_retrieve(monkeypatch):
    """替换检索:返回固定分数,记录调用次数。"""

    state = {"calls": 0, "scores": (0.7,)}

    async def fake_retrieve(**kwargs):
        state["calls"] += 1
        return _hits(list(state["scores"]))

    monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)
    return state


async def test_chitchat_uses_fixed_reply_and_zero_model_calls(patch_retrieve):
    deps, model, _ = _deps(intent="闲聊")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "你好"}, {"configurable": {"thread_id": "t1"}})
    assert out["route"] == "chitchat"
    assert out["reply"] == CHITCHAT_REPLY
    assert out["finish_reason"] == "chitchat"
    assert out["actions"] == []
    assert model.calls == 0  # 闲聊一次模型都不调
    assert patch_retrieve["calls"] == 0


async def test_complaint_returns_two_independent_actions(patch_retrieve):
    deps, model, _ = _deps(intent="投诉")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "我要投诉"}, {"configurable": {"thread_id": "t1"}})
    assert out["reply"] == COMPLAINT_REPLY
    assert [a["id"] for a in out["actions"]] == ["transfer_human", "create_ticket"]
    assert model.calls == 0
    assert patch_retrieve["calls"] == 0


async def test_business_route_goes_straight_to_agent(patch_retrieve):
    deps, model, _ = _deps(intent="物流")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "订单1001的物流到哪了"}, {"configurable": {"thread_id": "t1"}})
    assert out["route"] == "business"
    assert out["finish_reason"] == "stop"
    assert model.calls == 1
    assert patch_retrieve["calls"] == 0  # 业务数据类不预检索
    assert out.get("gate_passed") is None  # 没检索就没有闸这一说


async def test_knowledge_route_must_retrieve_then_pass_gate(patch_retrieve):
    """验收标准 1:政策类问题强制走检索节点。"""
    patch_retrieve["scores"] = (0.72,)
    deps, model, check = _deps(intent="商品咨询")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "退货运费谁出"}, {"configurable": {"thread_id": "t1"}})
    assert out["route"] == "knowledge"
    assert patch_retrieve["calls"] == 1
    assert check.calls == 1
    assert out["gate_passed"] is True
    assert out["finish_reason"] == "stop"
    assert "运费" in out["reply"] or model.calls == 1


async def test_knowledge_route_abstains_when_score_low(patch_retrieve):
    patch_retrieve["scores"] = (0.39,)
    deps, model, check = _deps(intent="退款退货")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "帮我写首诗"}, {"configurable": {"thread_id": "t1"}})
    assert out["gate_passed"] is False
    assert out["gate_reason"] == "score_low"
    assert out["reply"] == ABSTAIN_REPLY
    assert out["finish_reason"] == "abstain"
    assert model.calls == 0  # 闸挡住就不进 Agent
    assert [a["id"] for a in out["actions"]] == ["transfer_human", "create_ticket"]


async def test_knowledge_route_abstains_when_selfcheck_says_no(patch_retrieve):
    deps, model, check = _deps(intent="商品咨询", useful=False)
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "W40 楦宽多少毫米"}, {"configurable": {"thread_id": "t1"}})
    assert out["gate_reason"] == "self_check"
    assert model.calls == 0


async def test_intent_fallback_keeps_graph_moving(patch_retrieve):
    """意图模型挂了也要有答复,不能整轮静默。"""

    class BrokenIntent:
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("intent upstream 500")

    deps, model, _ = _deps()
    deps.intent_model = BrokenIntent()
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "随便说点什么"}, {"configurable": {"thread_id": "t1"}})
    assert out["intent_fallback"] is True
    assert out["intent"] == "售后"
    assert out["finish_reason"] == "stop"


async def test_trace_node_records_structured_summary(patch_retrieve):
    deps, _, _ = _deps(intent="投诉")
    app = build_agent_graph(deps)
    out = await app.ainvoke({"user_input": "我要投诉"}, {"configurable": {"thread_id": "t1"}})
    assert out["trace"]["intent"] == "投诉"
    assert out["trace"]["route"] == "complaint"
    assert out["trace"]["actions"] == ["transfer_human", "create_ticket"]


async def test_same_thread_keeps_state_across_turns(patch_retrieve):
    """checkpointer 的意义:同一 thread 的第二轮能看见上一轮的状态。"""
    deps, _, _ = _deps(intent="闲聊")
    app = build_agent_graph(deps)
    cfg = {"configurable": {"thread_id": "same"}}
    await app.ainvoke({"user_input": "你好"}, cfg)
    await app.ainvoke({"user_input": "谢谢"}, cfg)
    snapshot = await app.aget_state(cfg)
    assert snapshot.values["intent"] == "闲聊"
    assert snapshot.values["user_input"] == "谢谢"


async def test_different_threads_are_isolated(patch_retrieve):
    deps, _, _ = _deps(intent="闲聊")
    app = build_agent_graph(deps)
    await app.ainvoke({"user_input": "你好"}, {"configurable": {"thread_id": "a"}})
    snapshot = await app.aget_state({"configurable": {"thread_id": "b"}})
    assert snapshot.values == {}
