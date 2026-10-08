import asyncio
import json
import time

from langchain_core.tools import tool
from pydantic import BaseModel

from customer_service.tool_executor import ToolExecutor


class KeywordInput(BaseModel):
    keyword: str


def test_executor_validates_retries_and_preserves_call_id():
    attempts = 0

    @tool(args_schema=KeywordInput)
    def query_faq(keyword: str) -> str:
        """Look up a FAQ keyword."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("slow")
        return keyword

    executor = ToolExecutor([query_faq], timeout_seconds=0.1)

    async def run():
        unknown = await executor.execute({"id": "x", "name": "unknown", "args": {}})
        assert unknown.status == "error" and unknown.tool_call_id == "x"
        invalid = await executor.execute({"id": "y", "name": "query_faq", "args": {}})
        assert invalid.status == "error" and attempts == 0
        ok = await executor.execute({"id": "z", "name": "query_faq", "args": {"keyword": "退货"}})
        assert ok.content == "退货" and ok.tool_call_id == "z" and attempts == 2
    asyncio.run(run())


def test_ticket_does_not_retry():
    attempts = 0

    @tool(args_schema=KeywordInput)
    def create_ticket(keyword: str) -> str:
        """Create a ticket."""
        nonlocal attempts
        attempts += 1
        raise RuntimeError("private detail")

    async def run():
        result = await ToolExecutor([create_ticket]).execute({"id": "c", "name": "create_ticket", "args": {"keyword": "x"}})
        assert result.status == "error"
        assert "private detail" not in result.content
        assert attempts == 1
    asyncio.run(run())
