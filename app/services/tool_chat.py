"""带 Function Calling 的流式客服对话(spec D4 单轮工具调用架构)。

数据流(spec §3):
  读历史(messages 表)→ bind_tools invoke 定工具 → tool_start 状态帧 →
  注册中心调度执行 → tool_end 状态帧 → ToolMessage 回灌 → astream 收敛逐 token
  → 全量落库(user / assistant(tool_calls) / tool / assistant 终答)。

单轮收敛定义(spec §3):第一次 invoke 产 tool_calls 则执行并回灌后再调一次;
第二次仍产 tool_calls(GLM 偶发),以工具结果组织兜底答复,不再执行——
保证「只做单轮调用」且不循环。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import build_chat_prompt_with_tools
from app.core.session import trim_history
from app.core.tools.business_tools import build_default_registry
from app.core.tools.context import CitationBucket, current_citations, current_conversation_id
from app.core.tools.executor import ToolExecutor, ToolResult
from app.core.tools.registry import ToolRegistry
from app.models.message import Message
from app.rag.citations import format_evidence
from app.rag.selfcheck import ABSTAIN_REPLY, check_evidence
from app.repositories import conversation_repo, low_confidence_repo, message_repo

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_LENGTH = 200
MAX_TOOL_CONTENT_LENGTH = 2000  # 落库的工具结果截断(回灌模型的全文由 ToolMessage 承载)

# Per-session lock: 同会话请求串行化,读历史→流式→落库不交错(CH01 review #2 的延续)。
_session_locks: dict[str, asyncio.Lock] = {}


def _get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


def _sanitize(message: str) -> str:
    """Never leak the API key through upstream error text (CH01 review Minor #9)。"""
    key = get_settings().llm_api_key
    if key and key in message:
        message = message.replace(key, "***")
    return message[:MAX_ERROR_MESSAGE_LENGTH]


def _to_langchain(msgs: list[Message]) -> list[BaseMessage]:
    """messages 表记录 → LangChain 消息,按 Chat Completions 协议重建工具轨迹。

    assistant(tool_calls) 与 tool(tool_call_id) 两条对号入座(spec Review Focus 2)。
    配对清洗(review C1):带工具轮次若被中断(用户关页面/上游异常),落库可能是
    [user, assistant(tool_calls)](tool 消息未落)或结尾的 tool 消息被 trim 裁掉。
    此时若原样回灌,assistant 的 tool_calls 没有对应 tool 消息,GLM/OpenAI 端点
    直接 HTTP 400,且该会话此后每轮都失败(会话被永久毒化)。
    处置:配不齐的 assistant(tool_calls) 降级为普通 AIMessage(丢掉 tool_calls),
    协议合法、内容不丢。
    """
    out: list[BaseMessage] = []
    for i, m in enumerate(msgs):
        if m.role == "user":
            out.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            if m.tool_calls:
                paired = _tool_calls_paired(msgs, i)
                if paired:
                    out.append(AIMessage(content=m.content or "", tool_calls=m.tool_calls))
                else:
                    # 降级:丢掉 tool_calls,保留正文(通常为空)。
                    out.append(AIMessage(content=m.content or ""))
            else:
                out.append(AIMessage(content=m.content or ""))
        else:  # tool
            out.append(ToolMessage(content=m.content or "", tool_call_id=m.tool_call_id or ""))
    return out


def _tool_calls_paired(msgs: list[Message], assistant_idx: int) -> bool:
    """assistant(tool_calls) 之后是否紧跟覆盖全部 call id 的 tool 消息。"""
    call_ids = {tc.get("id") for tc in msgs[assistant_idx].tool_calls or []}
    if not call_ids:
        return False
    following: set[str] = set()
    for m in msgs[assistant_idx + 1 :]:
        if m.role != "tool":
            break
        if m.tool_call_id:
            following.add(m.tool_call_id)
    return call_ids <= following


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(data=data, event=event)


async def chat_with_tools(
    session_id: str,
    message: str,
    conversation_id: int | None = None,
    *,
    new_conversation: bool = False,
    session: AsyncSession,
    model: ChatOpenAI | None = None,
    registry: ToolRegistry | None = None,
    executor: ToolExecutor | None = None,
    check_model=None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield SSE events(spec D7):
    tool_start → token… → tool_end → token… → citations → done,或单条 error。
    done 事件带 conversation_id,前端据此续聊;new_conversation=True 强制开新会话。

    ch04 生成质量控制插在工具执行之后、生成之前:
    - 检索零命中 → 拒答(入池 retrieval_low_conf);
    - 证据自评判不够 → 拒答(入池 self_check);
    两条都不进生成,避免先流式吐一段再道歉。
    citations 事件下发引用编号与原文,前端渲染可点角标。
    """
    model = model if model is not None else get_chat_model()
    registry = registry if registry is not None else build_default_registry()
    executor = executor if executor is not None else ToolExecutor(registry)
    bound = model.bind_tools(registry.for_bind())

    async def persist(conversation_id: int, **kwargs) -> str | None:
        """落库;失败返回脱敏错误信息(不抛)。review I4:落库阶段的 DB 异常
        (死连接/锁等待)也必须 yield error 收尾,不能让流裸奔中断。"""
        try:
            await message_repo.add_message(session, conversation_id, **kwargs)
            return None
        except Exception as exc:
            return _sanitize(str(exc))

    async with _get_lock(session_id):
        try:
            conv = await conversation_repo.get_or_create(
                session, conversation_id, session_id, force_new=new_conversation
            )
        except Exception as exc:
            yield _sse("error", {"code": "DB_ERROR", "message": _sanitize(str(exc))})
            return

        token_ctx = current_conversation_id.set(conv.id)
        citation_ctx = current_citations.set(CitationBucket())  # 每轮独立桶,避免串味
        try:
            # 历史先读、本轮后落:history 不含当前输入(与 CH01 prompt {input} 分工一致)。
            try:
                history = trim_history(
                    _to_langchain(await message_repo.list_by_conversation(session, conv.id)),
                    get_settings().history_token_budget,
                )
                prompt_messages = build_chat_prompt_with_tools().format_messages(
                    messages=history, input=message
                )
            except Exception as exc:  # 预流式失败也走 error 事件,连接不挂起
                yield _sse("error", {"code": "PROMPT_ERROR", "message": _sanitize(str(exc))})
                return
            # 1. 模型定工具(非流式,一次调用拿完整 tool_calls)。
            try:
                first = await bound.ainvoke(prompt_messages)
            except Exception as exc:
                yield _sse("error", {"code": "UPSTREAM_ERROR", "message": _sanitize(str(exc))})
                return
            # 上游出错不落半截(CH01 语义):user 消息在拿到响应后才落库。
            err = await persist(conv.id, role="user", content=message)
            if err:
                yield _sse("error", {"code": "DB_ERROR", "message": err})
                return

            # 2. 有工具调用:状态帧 → 执行 → 回灌。
            tool_messages: list[ToolMessage] = []
            tool_results: list[ToolResult] = []
            if first.tool_calls:
                err = await persist(
                    conv.id,
                    role="assistant",
                    content=first.content or None,
                    tool_calls=[
                        {
                            "name": tc["name"],
                            "args": tc["args"],
                            "id": tc["id"],
                            "type": "tool_call",
                        }
                        for tc in first.tool_calls
                    ],
                )
                if err:
                    yield _sse("error", {"code": "DB_ERROR", "message": err})
                    return
                for tc in first.tool_calls:
                    yield _sse(
                        "tool_start",
                        {"tool_call_id": tc["id"], "name": tc["name"], "args": tc["args"]},
                    )
                    result = await executor.execute(tc)
                    tool_results.append(result)
                    yield _sse(
                        "tool_end",
                        {
                            "tool_call_id": result.tool_call_id,
                            "name": result.name,
                            "ok": result.ok,
                            "summary": result.content[:MAX_TOOL_CONTENT_LENGTH],
                        },
                    )
                    tool_messages.append(
                        ToolMessage(content=result.content, tool_call_id=result.tool_call_id)
                    )
                    err = await persist(
                        conv.id,
                        role="tool",
                        content=result.content[:MAX_TOOL_CONTENT_LENGTH],
                        tool_call_id=result.tool_call_id,
                    )
                    if err:
                        yield _sse("error", {"code": "DB_ERROR", "message": err})
                        return

            # 2.5 生成前的质量闸(ch04):检索零命中 / 证据自评不够 → 拒答并入池。
            # 闸口必须在生成之前:答复是流式的,吐出去再判就只剩事后道歉了。
            bucket = current_citations.get() or CitationBucket()
            citations = bucket.items
            if bucket.called:
                verdict = None
                if citations and get_settings().evidence_selfcheck_enabled:
                    verdict = await check_evidence(
                        message, format_evidence(citations), model=check_model
                    )
                if not citations or (verdict is not None and not verdict.useful):
                    source = "self_check" if citations else "retrieval_low_conf"
                    reason = verdict.reason if verdict is not None else "检索零命中"
                    try:
                        await low_confidence_repo.add_question(
                            session,
                            raw_question=message,
                            source=source,
                            reason=reason or None,
                            conversation_id=conv.id,
                        )
                    except Exception as exc:  # noqa: BLE001 - 入池失败不影响拒答
                        logger.warning("低置信度问题入池失败:%s", exc)
                    err = await persist(conv.id, role="assistant", content=ABSTAIN_REPLY)
                    if err:
                        yield _sse("error", {"code": "DB_ERROR", "message": err})
                        return
                    yield _sse("token", {"content": ABSTAIN_REPLY})
                    yield _sse(
                        "done",
                        {
                            "session_id": session_id,
                            "conversation_id": conv.id,
                            "finish_reason": "abstain",
                            "abstain_source": source,
                        },
                    )
                    return

            # 3. 收敛:流式吐终答;二次仍要工具 → 兜底摘要,不再执行(spec §3)。
            messages = prompt_messages + [first] + tool_messages
            finish_reason = "stop"
            reply_parts: list[str] = []
            saw_tool_calls = False
            try:
                async for chunk in bound.astream(messages):
                    if chunk.tool_call_chunks:
                        saw_tool_calls = True
                        continue  # 工具申请块不推给用户
                    text = chunk.text
                    if not text:
                        # GLM 思考分块无 content,跳过(CH01 spec D6)。
                        continue
                    reply_parts.append(text)
                    chunk_finish = (chunk.response_metadata or {}).get("finish_reason")
                    if chunk_finish:
                        finish_reason = chunk_finish
                    yield _sse("token", {"content": text})
            except Exception as exc:
                yield _sse("error", {"code": "UPSTREAM_ERROR", "message": _sanitize(str(exc))})
                return

            if saw_tool_calls and not reply_parts:
                # 兜底(review Minor 4):模型二次又要调工具,不执行,把已查到的
                # 结果包装成一句人话给用户,不直出原始工具文本。
                raw = " ".join(r.content for r in tool_results)[:MAX_TOOL_CONTENT_LENGTH]
                fallback = (
                    f"已为您查到以下信息：{raw}"
                    if raw
                    else "这个问题我暂时无法处理，请您稍后再试，或把情况描述得更具体一些。"
                )
                reply_parts.append(fallback)
                yield _sse("token", {"content": fallback})

            err = await persist(conv.id, role="assistant", content="".join(reply_parts))
            if err:
                yield _sse("error", {"code": "DB_ERROR", "message": err})
                return
            if citations:
                # 引用编号与原文随流下发:前端渲染可点角标,点开看来源章节与原文
                yield _sse(
                    "citations",
                    {"citations": [c.to_payload() for c in citations]},
                )
            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "conversation_id": conv.id,
                    "finish_reason": finish_reason,
                },
            )
        finally:
            current_citations.reset(citation_ctx)
            current_conversation_id.reset(token_ctx)
