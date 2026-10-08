import asyncio
from types import MappingProxyType

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError


class ToolExecutor:
    def __init__(self, tools: list[BaseTool], timeout_seconds: float = 5, faq_timeout_seconds: float | None = None) -> None:
        self.registry = MappingProxyType({item.name: item for item in tools})
        self.timeout_seconds = timeout_seconds
        self.faq_timeout_seconds = faq_timeout_seconds

    async def execute(self, call: dict) -> ToolMessage:
        name = call.get("name", "")
        call_id = call.get("id", "")
        item = self.registry.get(name)
        if item is None:
            return ToolMessage(content="未知工具，无法执行。", tool_call_id=call_id, name=name or "unknown", status="error")
        try:
            args = item.args_schema.model_validate(call.get("args", {})).model_dump()
        except (ValidationError, TypeError, ValueError):
            return ToolMessage(content="工具参数不符合要求。", tool_call_id=call_id, name=name, status="error")
        tries = 1 if name == "create_ticket" or (name == "query_faq" and self.faq_timeout_seconds is not None) else 2
        for attempt in range(tries):
            try:
                timeout = self.faq_timeout_seconds if name == "query_faq" and self.faq_timeout_seconds is not None else self.timeout_seconds
                content = await asyncio.wait_for(asyncio.to_thread(item.invoke, args), timeout=timeout)
                return ToolMessage(content=str(content), tool_call_id=call_id, name=name)
            except (TimeoutError, asyncio.TimeoutError):
                if attempt + 1 < tries:
                    continue
                return ToolMessage(content="工具查询超时，请稍后重试。", tool_call_id=call_id, name=name, status="error")
            except Exception:
                return ToolMessage(content="工具执行失败，请稍后重试。", tool_call_id=call_id, name=name, status="error")
        raise AssertionError("unreachable")
