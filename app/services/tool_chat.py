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
from app.core.tools.context import current_conversation_id
from app.core.tools.executor import ToolExecutor, ToolResult
from app.core.tools.registry import ToolRegistry
from app.models.message import Message
from app.repositories import conversation_repo, message_repo

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
    """
    out: list[BaseMessage] = []
    for m in msgs:
        if m.role == "user":
            out.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            if m.tool_calls:
                out.append(AIMessage(content=m.content or "", tool_calls=m.tool_calls))
            else:
                out.append(AIMessage(content=m.content or ""))
        else:  # tool
            out.append(ToolMessage(content=m.content or "", tool_call_id=m.tool_call_id or ""))
    return out


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(data=data, event=event)


async def chat_with_tools(
    session_id: str,
    message: str,
    conversation_id: int | None = None,
    *,
    session: AsyncSession,
    model: ChatOpenAI | None = None,
    registry: ToolRegistry | None = None,
    executor: ToolExecutor | None = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield SSE events(spec D7):
    tool_start → token… → tool_end → token… → done,或单条 error。
    done 事件带 conversation_id,前端据此续聊。
    """
    model = model if model is not None else get_chat_model()
    registry = registry if registry is not None else build_default_registry()
    executor = executor if executor is not None else ToolExecutor(registry)
    bound = model.bind_tools(registry.for_bind())

    async with _get_lock(session_id):
        try:
            conv = await conversation_repo.get_or_create(session, conversation_id, session_id)
        except Exception as exc:
            yield _sse("error", {"code": "DB_ERROR", "message": _sanitize(str(exc))})
            return

        token_ctx = current_conversation_id.set(conv.id)
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
            await message_repo.add_message(session, conv.id, "user", message)

            # 2. 有工具调用:状态帧 → 执行 → 回灌。
            tool_messages: list[ToolMessage] = []
            tool_results: list[ToolResult] = []
            if first.tool_calls:
                await message_repo.add_message(
                    session,
                    conv.id,
                    "assistant",
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
                    await message_repo.add_message(
                        session,
                        conv.id,
                        "tool",
                        content=result.content[:MAX_TOOL_CONTENT_LENGTH],
                        tool_call_id=result.tool_call_id,
                    )

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
                # 兜底:模型二次又要调工具,不执行,直接把已查到的结果组织给它。
                fallback = " ".join(r.content for r in tool_results)[:MAX_TOOL_CONTENT_LENGTH]
                if fallback:
                    reply_parts.append(fallback)
                    yield _sse("token", {"content": fallback})

            await message_repo.add_message(
                session, conv.id, "assistant", content="".join(reply_parts)
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
            current_conversation_id.reset(token_ctx)
