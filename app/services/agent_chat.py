"""图编排版对话服务:图负责想清楚,这里负责说话和记账。

职责边界很清楚,不能混:
- **图**只读写 State,不碰数据库、不知道 SSE 是什么;
- **这一层**管会话(取/建)、历史(读/裁)、落库(user / 工具轨迹 / 终答)、
  把图吐出来的事件翻译成 SSE 帧。

这样图能单独 ainvoke 测,服务层能拿假图测,两边都不必背着对方跑。

事件契约(与 CH02 一致,只增不改):
  node → token… → tool_start → tool_end … → citations → done
新增:`node`(节点进出)、`intent`(判意图结果)、`gate`(闸的结论),
`done` 多带 actions / intent / route。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.gate import REASON_NO_HITS, REASON_SCORE_LOW
from app.agent.graph import AgentDeps, build_agent_graph
from app.config import get_settings
from app.core.session import trim_history
from app.core.tools.business_tools import build_default_registry
from app.core.tools.context import CitationBucket, current_citations, current_conversation_id
from app.models.message import Message
from app.repositories import conversation_repo, low_confidence_repo, message_repo
from app.services.tool_chat import _get_lock, _sanitize, _to_langchain

logger = logging.getLogger(__name__)

MAX_TOOL_CONTENT_LENGTH = 2000

# 每轮开跑前显式重置的本轮字段。
#
# checkpointer 以 thread_id = conversation_id 保留 State,下一轮只更新输入那几项,
# 其余字段原样留在上一次的值上。于是上一轮的 `order_options`、`finish_reason`
# 会漏到本轮的 done 事件里——第六章联调时真实发生过:本轮闸没过走了拒答,
# 前端却既收到拒答话术又收到上一轮的订单卡片。
#
# 放在调用点而不是每个节点里:节点只对自己写的字段负责,"一轮从哪开始"是服务层
# 的知识。要新增本轮字段,加到这张表里,别指望节点挨个清。
_TURN_RESET: dict = {
    "canonical_query": "",
    "resolved_query": "",
    "understanding_changed": False,
    "understanding_fallback": False,
    "slot_order_id": "",
    "intent": "",
    "intent_confidence": 0.0,
    "intent_fallback": False,
    "route": "",
    "order_facts": "",
    "order_options": [],
    "route_queries": [],
    "citations": [],
    "top_score": 0.0,
    "gate_passed": None,
    "gate_reason": "",
    "gate_detail": "",
    "reply": "",
    "finish_reason": "",
    "actions": [],
    "trail": [],
    "steps": 0,
    "tokens": 0,
    "stopped_by": "",
    "trace": {},
}


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(data=data, event=event)


def _unpack(chunk) -> tuple[str, dict] | None:
    """custom stream 的返回形态随调用方式/版本变,三种都认:
    单选一个 stream_mode → 裸 payload 字典;传列表 → ("custom", payload) 元组;
    部分版本 → {"type","data"} 字典。升一次库挂一次的成本太高,这里一次兜住。"""
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[1], dict):
        return chunk[0], chunk[1]
    if isinstance(chunk, dict):
        if "type" in chunk and isinstance(chunk.get("data"), dict):
            return chunk["type"], chunk["data"]
        return "custom", chunk
    return None


def _custom_to_sse(data: dict) -> ServerSentEvent | None:
    kind = data.get("kind")
    if kind == "token":
        return _sse("token", {"content": data.get("text", "")})
    if kind == "tool_start":
        return _sse(
            "tool_start",
            {"tool_call_id": data.get("tool_call_id"), "name": data.get("name"), "args": data.get("args") or {}},
        )
    if kind == "tool_end":
        return _sse(
            "tool_end",
            {
                "tool_call_id": data.get("tool_call_id"),
                "name": data.get("name"),
                "ok": data.get("ok"),
                "summary": str(data.get("summary") or "")[:MAX_TOOL_CONTENT_LENGTH],
            },
        )
    if kind == "node":
        return _sse("node", {"node": data.get("node"), "phase": data.get("phase")})
    if kind == "intent":
        return _sse(
            "intent",
            {"intent": data.get("intent"), "route": data.get("route"), "confidence": data.get("confidence")},
        )
    if kind == "order_options":
        return _sse("order_options", {"orders": data.get("orders") or []})
    if kind == "resolved":
        return _sse(
            "resolved",
            {
                "resolved": data.get("resolved", ""),
                "canonical": data.get("canonical", ""),
                "changed": bool(data.get("changed")),
                "order_id": data.get("order_id") or None,
            },
        )
    if kind == "gate":
        return _sse(
            "gate",
            {
                "passed": data.get("passed"),
                "reason": data.get("reason", ""),
                "top_score": data.get("top_score", 0.0),
            },
        )
    return None


def build_default_deps(session_factory) -> AgentDeps:
    """生产依赖:真模型 + 默认工具注册表 + 默认检索器 + 内存 checkpointer。"""
    from app.agent.checkpointer import get_checkpointer
    from app.core.llm import get_chat_model, get_extract_model

    settings = get_settings()
    return AgentDeps(
        model=get_chat_model(),
        registry=build_default_registry(),
        session_factory=session_factory,
        intent_model=get_extract_model(),
        check_model=get_extract_model(),
        understand_model=get_extract_model(),
        expand_model=get_extract_model(),
        kb_min_score=settings.kb_min_score,
        max_steps=settings.agent_max_steps,
        token_budget=settings.agent_token_budget,
        checkpointer=get_checkpointer(),
    )


async def _persist_trail(session: AsyncSession, conversation_id: int, trail: list) -> str | None:
    """把 Agent 的工具轨迹按协议落库:assistant(tool_calls) + tool 逐条对号。"""
    for msg in trail:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            await message_repo.add_message(
                session,
                conversation_id,
                role="assistant",
                content=str(msg.content or "") or None,
                tool_calls=[
                    {"name": tc["name"], "args": tc.get("args") or {}, "id": tc["id"], "type": "tool_call"}
                    for tc in msg.tool_calls
                ],
            )
        else:
            await message_repo.add_message(
                session,
                conversation_id,
                role="tool",
                content=str(msg.content or "")[:MAX_TOOL_CONTENT_LENGTH],
                tool_call_id=getattr(msg, "tool_call_id", None),
            )
    return None


async def chat_with_agent(
    session_id: str,
    message: str,
    conversation_id: int | None = None,
    *,
    new_conversation: bool = False,
    order_id: str | None = None,
    session: AsyncSession,
    deps: AgentDeps | None = None,
    graph=None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """跑一轮:建/取会话 → 读历史 → 图执行(边跑边推事件)→ 落库 → 下发结果。"""
    from app.db import get_sessionmaker

    deps = deps or build_default_deps(get_sessionmaker())
    app_graph = graph or build_agent_graph(deps)

    async with _get_lock(session_id):
        try:
            conv = await conversation_repo.get_or_create(
                session, conversation_id, session_id, force_new=new_conversation
            )
        except Exception as exc:
            yield _sse("error", {"code": "DB_ERROR", "message": _sanitize(str(exc))})
            return

        token_ctx = current_conversation_id.set(conv.id)
        citation_ctx = current_citations.set(CitationBucket())
        try:
            try:
                history = trim_history(
                    _to_langchain(await message_repo.list_by_conversation(session, conv.id)),
                    get_settings().history_token_budget,
                )
            except Exception as exc:
                yield _sse("error", {"code": "PROMPT_ERROR", "message": _sanitize(str(exc))})
                return

            # 上游真跑起来之后才落 user:失败不留半截(CH01 语义)
            final_state: dict = {}
            try:
                async for chunk in app_graph.astream(
                    {
                        **_TURN_RESET,
                        "user_input": message,
                        "history": history,
                        "user_key": session_id,
                        "order_id": order_id or "",
                    },
                    {"configurable": {"thread_id": str(conv.id)}},
                    stream_mode="custom",
                    version="v2",
                ):
                    unpacked = _unpack(chunk)
                    if unpacked is None:
                        continue
                    mode, data = unpacked
                    if mode != "custom":
                        continue
                    frame = _custom_to_sse(data)
                    if frame is not None:
                        yield frame
                snapshot = await app_graph.aget_state({"configurable": {"thread_id": str(conv.id)}})
                final_state = dict(snapshot.values)
            except Exception as exc:
                logger.exception("agent graph failed")
                yield _sse("error", {"code": "UPSTREAM_ERROR", "message": _sanitize(str(exc))})
                return

            err = await _persist_turn(session, conv.id, message, final_state)
            if err:
                yield _sse("error", {"code": "DB_ERROR", "message": err})
                return

            citations = final_state.get("citations") or []
            if not citations:
                # 业务数据类没走预检索,但 Agent 自己调了 query_faq:引用走旁路拿回来
                bucket = current_citations.get()
                if bucket is not None and bucket.items:
                    citations = [c.to_payload() for c in bucket.items]
            if citations:
                yield _sse("citations", {"citations": citations})

            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "conversation_id": conv.id,
                    "finish_reason": final_state.get("finish_reason") or "stop",
                    "intent": final_state.get("intent"),
                    "route": final_state.get("route"),
                    "actions": final_state.get("actions") or [],
                    "order_options": final_state.get("order_options") or [],
                    "gate": {
                        "passed": final_state.get("gate_passed"),
                        "reason": final_state.get("gate_reason") or "",
                    },
                },
            )
        finally:
            current_citations.reset(citation_ctx)
            current_conversation_id.reset(token_ctx)


async def _persist_turn(session: AsyncSession, conversation_id: int, message: str, state: dict) -> str | None:
    """user → 工具轨迹 → 终答,顺序与协议一致;DB 异常转成脱敏文本返回,不抛。"""
    try:
        await message_repo.add_message(session, conversation_id, role="user", content=message)
        trail = state.get("trail") or []
        if trail:
            await _persist_trail(session, conversation_id, trail)
        reply = state.get("reply") or ""
        if reply:
            await message_repo.add_message(session, conversation_id, role="assistant", content=reply)
        if state.get("finish_reason") == "abstain":
            reason = state.get("gate_reason") or ""
            source = "retrieval_low_conf" if reason in (REASON_NO_HITS, REASON_SCORE_LOW) else "self_check"
            detail = state.get("gate_detail") or ""
            try:
                await low_confidence_repo.add_question(
                    session,
                    raw_question=state.get("resolved_query") or message,
                    source=source,
                    reason=(f"{reason}: {detail}"[:255] if detail else reason or None),
                    conversation_id=conversation_id,
                )
            except Exception as exc:  # noqa: BLE001 - 入池失败不影响答复
                logger.warning("低置信度问题入池失败:%s", exc)
        return None
    except Exception as exc:
        return _sanitize(str(exc))
