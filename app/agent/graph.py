"""客服工作流的图骨架(第五章第 2 点)。

    START → 指代消解 → 意图识别 ─┬─ 知识类 → 检索 → 置信度闸 ─┬─ 过闸 → 主力 Agent ─┐
                                 │                              └─ 不过 → 兜底话术 ─┤
                                 ├─ 业务数据类 → 主力 Agent ───────────────────────┤
                                 ├─ 投诉 → 安抚 + 两个选项 ───────────────────────┤
                                 └─ 闲聊 → 固定话术 ───────────────────────────────┘
                                                                                   ↓
                                                                              日志记录 → END

分工很明确:**能确定的事写进图**(分流、检索、过闸、兜底),**不确定的事才交给模型**
(判意图、决定调哪个工具、组织语言)。图是骨架,Agent 是骨架里的一个节点——不是反过来。

节点保持纯函数:不碰数据库、不碰 SSE。它们只读写 State,把要推给前端的东西扔进
`get_stream_writer()`,把要落库的东西留在 State 里让服务层去写。这样每个节点都能
单独 ainvoke 测,不必先起一个服务。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.gate import check_gate, citations_from_payload
from app.agent.intent import (
    ROUTE_BUSINESS,
    ROUTE_CHITCHAT,
    ROUTE_COMPLAINT,
    ROUTE_KNOWLEDGE,
    classify_intent,
)
from app.agent.react import run_react
from app.rag.citations import build_citations, format_evidence
from app.rag.retriever import retrieve
from app.rag.selfcheck import ABSTAIN_REPLY

logger = logging.getLogger(__name__)

# 知识类检索取多少条证据送进 Agent(ch04 的 Top-10 口径)
EVIDENCE_TOP_K = 10

COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验,您说的情况我记下了。"
    "这个问题需要专人跟进,您可以选择转人工客服直接沟通,"
    "或者我帮您建一张工单转给售后团队处理——两个都可以选,也可以都选。"
)

CHITCHAT_REPLY = (
    "您好呀,我是小智,电商平台的在线客服。"
    "查订单、看物流、退换货、问商品参数都可以找我,您想了解什么?"
)

# 两个动作互不绑定:前端各渲染一个按钮,后端只给建议、不自动执行(第 8 点)
ACTION_TRANSFER = {"id": "transfer_human", "label": "转人工"}
ACTION_TICKET = {"id": "create_ticket", "label": "建工单"}
ESCALATION_ACTIONS = [ACTION_TRANSFER, ACTION_TICKET]


class AgentState(TypedDict, total=False):
    """一轮对话的完整状态。只装这一轮需要的东西,历史靠 MySQL。"""

    # 输入
    user_input: str
    history: list[BaseMessage]  # 服务层注入的裁剪后历史(每轮覆盖,不用 reducer)
    # 分流
    resolved_query: str
    intent: str
    intent_confidence: float
    intent_fallback: bool
    route: str
    # 检索与闸
    citations: list[dict]
    top_score: float
    gate_passed: bool
    gate_reason: str
    gate_detail: str
    # 产物
    reply: str
    finish_reason: str
    actions: list[dict]
    trail: list[BaseMessage]
    steps: int
    tokens: int
    stopped_by: str
    trace: dict


@dataclass
class AgentDeps:
    """图要用到的外部依赖。全部可注入 = 单测能整张图跑假模型。"""

    model: object
    registry: object
    session_factory: object
    intent_model: object | None = None
    check_model: object | None = None
    retriever: object | None = None
    kb_min_score: float = 0.45
    max_steps: int = 4
    token_budget: int = 6000
    checkpointer: object = field(default_factory=InMemorySaver)


def _emit(payload: dict) -> None:
    """节点 → 外部的事件旁路。没有 writer(直接 ainvoke)时是空实现。"""
    writer = get_stream_writer()
    if writer is not None:
        writer(payload)


def _emit_node(name: str, phase: str) -> None:
    _emit({"kind": "node", "node": name, "phase": phase})


def _emit_token(text: str) -> None:
    if text:
        _emit({"kind": "token", "text": text})


def build_agent_graph(deps: AgentDeps):
    """编译图。deps 用闭包捕获——节点保持纯函数,不必知道依赖从哪来。"""

    # ------------------------------------------------------------ 节点
    async def resolve_reference(state: AgentState) -> dict:
        """指代消解:本章原样透传(CH06 换成真实现,节点位置不变)。"""
        _emit_node("resolve_reference", "enter")
        query = state.get("user_input", "")
        _emit_node("resolve_reference", "exit")
        return {"resolved_query": query}

    async def classify(state: AgentState) -> dict:
        _emit_node("classify_intent", "enter")
        result = await classify_intent(
            state.get("resolved_query", ""), model=deps.intent_model, history=state.get("history")
        )
        _emit_node("classify_intent", "exit")
        _emit({"kind": "intent", "intent": result.intent, "route": result.route, "confidence": result.confidence})
        return {
            "intent": result.intent,
            "route": result.route,
            "intent_confidence": result.confidence,
            "intent_fallback": result.fallback,
        }

    async def retrieve_knowledge(state: AgentState) -> dict:
        """知识类强制检索:这一步是确定性编排的核心,绕过它就没有证据。"""
        _emit_node("retrieve_knowledge", "enter")
        query = state.get("resolved_query", "")
        hits = []
        try:
            hits = await retrieve(
                session_factory=deps.session_factory,
                query=query,
                top_k=EVIDENCE_TOP_K,
                retriever=deps.retriever,
            )
        except Exception as exc:  # noqa: BLE001 - 检索挂了也要往下走,交给闸拒答
            logger.warning("知识类检索失败,按无证据处理:%s", exc)
        citations = build_citations(hits)
        _emit_node("retrieve_knowledge", "exit")
        return {
            "citations": [c.to_payload() for c in citations],
            "top_score": citations[0].score if citations else 0.0,
        }

    async def confidence_gate(state: AgentState) -> dict:
        """置信度闸:证据弱就到此为止,不进 Agent(答复是流式的,进去就晚了)。"""
        _emit_node("confidence_gate", "enter")
        citations = citations_from_payload(state.get("citations") or [])
        result = await check_gate(
            state.get("resolved_query", ""),
            citations,
            min_score=deps.kb_min_score,
            model=deps.check_model,
        )
        _emit_node("confidence_gate", "exit")
        _emit(
            {
                "kind": "gate",
                "passed": result.passed,
                "reason": result.reason,
                "top_score": round(result.top_score, 4),
            }
        )
        return {
            "gate_passed": result.passed,
            "gate_reason": result.reason,
            "gate_detail": result.detail,
        }

    async def agent(state: AgentState) -> dict:
        """主力 Agent:ReAct 循环,工具复用第 2 章那五个。"""
        _emit_node("agent", "enter")
        citations = citations_from_payload(state.get("citations") or [])
        evidence = format_evidence(citations) if citations else ""
        writer = get_stream_writer()
        outcome = await run_react(
            state.get("resolved_query", ""),
            model=deps.model,
            registry=deps.registry,
            history=state.get("history") or [],
            evidence=evidence,
            max_steps=deps.max_steps,
            token_budget=deps.token_budget,
            writer=writer,
        )
        _emit_node("agent", "exit")
        return {
            "reply": outcome.reply,
            "finish_reason": "stop",
            "trail": outcome.trail,
            "steps": outcome.model_calls,
            "tokens": outcome.tokens,
            "stopped_by": outcome.stopped_by,
            "actions": [],
        }

    async def abstain(state: AgentState) -> dict:
        """证据不够:固定兜底话术 + 两个出口。拒答不给下一步等于把人晾在原地。"""
        _emit_node("abstain", "enter")
        _emit_token(ABSTAIN_REPLY)
        _emit_node("abstain", "exit")
        return {
            "reply": ABSTAIN_REPLY,
            "finish_reason": "abstain",
            "actions": list(ESCALATION_ACTIONS),
            "gate_passed": False,
        }

    async def complaint_reply(state: AgentState) -> dict:
        """投诉:零模型调用。安抚 + 两个可选项,后端不替用户做决定。"""
        _emit_node("complaint_reply", "enter")
        _emit_token(COMPLAINT_REPLY)
        _emit_node("complaint_reply", "exit")
        return {
            "reply": COMPLAINT_REPLY,
            "finish_reason": "complaint",
            "actions": list(ESCALATION_ACTIONS),
        }

    async def chitchat_reply(state: AgentState) -> dict:
        """闲聊:固定话术,一次模型都不调——确定性编排最直接的价值。"""
        _emit_node("chitchat_reply", "enter")
        _emit_token(CHITCHAT_REPLY)
        _emit_node("chitchat_reply", "exit")
        return {"reply": CHITCHAT_REPLY, "finish_reason": "chitchat", "actions": []}

    async def log_trace(state: AgentState) -> dict:
        """日志记录节点:一行结构化日志,不建表(本章无新表)。"""
        _emit_node("log_trace", "enter")
        trace = {
            "intent": state.get("intent"),
            "route": state.get("route"),
            "intent_confidence": round(float(state.get("intent_confidence") or 0.0), 3),
            "intent_fallback": bool(state.get("intent_fallback")),
            "gate_passed": state.get("gate_passed"),
            "gate_reason": state.get("gate_reason") or "",
            "top_score": round(float(state.get("top_score") or 0.0), 4),
            "steps": state.get("steps", 0),
            "tokens": state.get("tokens", 0),
            "stopped_by": state.get("stopped_by", ""),
            "finish_reason": state.get("finish_reason", ""),
            "actions": [a["id"] for a in state.get("actions") or []],
        }
        logger.info("agent.trace %s", trace)
        _emit_node("log_trace", "exit")
        return {"trace": trace}

    # ------------------------------------------------------------ 条件边
    def route_after_intent(state: AgentState) -> str:
        route = state.get("route") or ROUTE_BUSINESS
        return {
            ROUTE_KNOWLEDGE: "retrieve_knowledge",
            ROUTE_COMPLAINT: "complaint_reply",
            ROUTE_CHITCHAT: "chitchat_reply",
            ROUTE_BUSINESS: "agent",
        }.get(route, "agent")

    def route_after_gate(state: AgentState) -> str:
        return "agent" if state.get("gate_passed") else "abstain"

    # ------------------------------------------------------------ 组装
    graph = StateGraph(AgentState)
    graph.add_node("resolve_reference", resolve_reference)
    graph.add_node("classify_intent", classify)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("confidence_gate", confidence_gate)
    graph.add_node("agent", agent)
    graph.add_node("abstain", abstain)
    graph.add_node("complaint_reply", complaint_reply)
    graph.add_node("chitchat_reply", chitchat_reply)
    graph.add_node("log_trace", log_trace)

    graph.add_edge(START, "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        ["retrieve_knowledge", "agent", "complaint_reply", "chitchat_reply"],
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges("confidence_gate", route_after_gate, ["agent", "abstain"])
    for terminal in ("agent", "abstain", "complaint_reply", "chitchat_reply"):
        graph.add_edge(terminal, "log_trace")
    graph.add_edge("log_trace", END)

    return graph.compile(checkpointer=deps.checkpointer)


@dataclass
class GraphRun:
    """一次图执行的结果 + 耗时,服务层拿它拼 SSE 的 done 事件。"""

    state: dict
    elapsed_ms: int = 0
    started: float = field(default_factory=time.perf_counter)

    def finish(self) -> "GraphRun":
        self.elapsed_ms = int((time.perf_counter() - self.started) * 1000)
        return self
