import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from random import Random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from customer_service.business_tools import build_business_tools
from customer_service.history import fit_history
from customer_service.prompts import CHAT_PROMPT, FINAL_RESPONSE_RULE
from customer_service.repository import MessageRow, Repository
from customer_service.tool_executor import ToolExecutor


@dataclass(frozen=True)
class ChatEvent:
    name: str
    data: dict


class ChatService:
    def __init__(self, model, repository: Repository, token_budget: int, knowledge_search=None) -> None:
        self.model = model
        self.repository = repository
        self.token_budget = token_budget
        self.knowledge_search = knowledge_search
        self._locks: dict[str, asyncio.Lock] = {}

    def prepare(self, conversation_id: str, message: str):
        if self.repository.get_conversation(conversation_id) is None:
            raise KeyError(conversation_id)
        rows = self.repository.list_messages(conversation_id)
        history = []
        for row in rows:
            if row.role == "user":
                history.append(HumanMessage(content=row.content))
            elif row.role == "assistant":
                history.append(AIMessage(content=row.content, tool_calls=row.tool_calls or []))
            elif row.role == "tool":
                history.append(ToolMessage(content=row.content, tool_call_id=row.tool_call_id or ""))
        final_rule_reserve = count_tokens_approximately([SystemMessage(content=FINAL_RESPONSE_RULE)])
        history = fit_history(history, message, self.token_budget - final_rule_reserve)
        return CHAT_PROMPT.format_messages(history=history, message=message)

    async def stream(self, conversation_id: str, message: str) -> AsyncIterator[ChatEvent]:
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            async for event in self._stream_locked(conversation_id, message):
                yield event

    async def _stream_locked(self, conversation_id: str, message: str) -> AsyncIterator[ChatEvent]:
        messages = await asyncio.to_thread(self.prepare, conversation_id, message)
        await asyncio.to_thread(self.repository.append_messages, conversation_id, [MessageRow("user", message)])
        tools = build_business_tools(self.repository, conversation_id, Random(), self.knowledge_search)
        decision = await self.model.bind_tools(tools, tool_choice="auto", parallel_tool_calls=False).ainvoke(messages)
        calls = decision.tool_calls
        if len(calls) > 1:
            raise ValueError("Model requested more than one tool")
        if calls:
            call = calls[0]
            yield ChatEvent("tool_status", {"tool_name": call["name"], "state": "running"})
            result = await ToolExecutor(tools, faq_timeout_seconds=30 if self.knowledge_search is not None else None).execute(call)
            original_result = result
            final_messages = [*messages, decision, result]
            max_input = self.token_budget - 512 - count_tokens_approximately([SystemMessage(content=FINAL_RESPONSE_RULE)])
            while count_tokens_approximately(final_messages) > max_input and len(final_messages) > 4:
                del final_messages[1]
                while len(final_messages) > 4 and not isinstance(final_messages[1], HumanMessage):
                    del final_messages[1]
            if count_tokens_approximately(final_messages) > max_input:
                result = ToolMessage(content="工具结果超过本轮上下文预算，无法引用详细结果。", tool_call_id=result.tool_call_id, name=call["name"], status="error")
                final_messages = [*final_messages[:-1], result]
            if count_tokens_approximately(final_messages) > max_input:
                raise ValueError("Tool result exceeds token budget")
            await asyncio.to_thread(self.repository.append_messages, conversation_id, [
                MessageRow("assistant", decision.content if isinstance(decision.content, str) else "", calls),
                MessageRow("tool", str(original_result.content), tool_call_id=original_result.tool_call_id),
            ])
            yield ChatEvent("tool_status", {"tool_name": call["name"], "state": "error" if result.status == "error" else "success"})
            messages = final_messages
        messages = [SystemMessage(content=f"{messages[0].content}\n{FINAL_RESPONSE_RULE}"), *messages[1:]]
        pieces = []
        async for chunk in self.model.astream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                pieces.append(chunk.content)
                yield ChatEvent("token", {"text": chunk.content})
        await asyncio.to_thread(self.repository.append_messages, conversation_id, [MessageRow("assistant", "".join(pieces))])
