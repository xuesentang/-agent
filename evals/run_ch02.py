"""Evaluate the three chapter-two prompts against a running local service."""

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from customer_service.config import Settings
from customer_service.db import make_engine
from customer_service.models import Ticket
from customer_service.repository import Repository


async def main() -> int:
    cases = [json.loads(line) for line in Path(__file__).with_name("ch02_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    engine = make_engine(Settings.from_env().database_url)
    repository = Repository(sessionmaker(engine, expire_on_commit=False))
    assert set(inspect(engine).get_table_names()) == {"faq", "conversations", "messages", "tickets"}
    assert repository.search_faq("邮费") == []
    failures = 0
    base_url = os.getenv("CH02_BASE_URL", "http://127.0.0.1:8000")
    async with httpx.AsyncClient(timeout=90) as client:
        for case in cases:
            response = await client.post(f"{base_url}/chat/stream", json={"message": case["message"]})
            events = []
            for block in response.text.strip().split("\n\n"):
                lines = block.splitlines()
                if len(lines) >= 2 and lines[0].startswith("event: "):
                    events.append((lines[0][7:], json.loads(lines[1][6:])))
            chosen = [data["tool_name"] for name, data in events if name == "tool_status" and data["state"] == "running"]
            answer = "".join(data["text"] for name, data in events if name == "token")
            cid = next((data["conversation_id"] for name, data in events if name == "session"), None)
            rows = repository.list_messages(cid) if cid else []
            tool_rows = [row for row in rows if row.role == "tool"]
            try:
                result = json.loads(tool_rows[0].content) if tool_rows else {}
            except json.JSONDecodeError:
                result = {}
            data_ok = True
            if case.get("expected_order_id"):
                data_ok = result.get("order_id") == case["expected_order_id"] and result.get("status", "") in answer
            if "expected_found" in case:
                data_ok = result.get("found") is case["expected_found"]
                tool_args = rows[1].tool_calls[0].get("args", {}) if len(rows) > 1 and rows[1].tool_calls else {}
                data_ok = data_ok and tool_args.get("keyword") == case["expected_keyword"]
                if case["expected_found"] is True:
                    data_ok = data_ok and "退货" in answer and ("条件" in answer or "政策" in answer)
                if case["expected_found"] is False:
                    data_ok = data_ok and bool(re.search(r"(?:未|没|没有).{0,16}(?:查到|找到)|暂无", answer))
            passed = response.status_code == 200 and chosen == [case["expected_tool"]] and bool(answer) and events[-1][0] == "done" and data_ok and [row.role for row in rows] == ["user", "assistant", "tool", "assistant"] and rows[1].tool_calls[0]["id"] == rows[2].tool_call_id
            print(json.dumps({"message": case["message"], "status": response.status_code, "tools": chosen, "tool_result": result, "answer": answer, "pass": passed}, ensure_ascii=False))
            failures += not passed
        response = await client.post(f"{base_url}/chat/stream", json={"message": "请帮我创建人工工单，我要转人工处理商品破损问题"})
        ticket_events = []
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            if len(lines) >= 2 and lines[0].startswith("event: "):
                ticket_events.append((lines[0][7:], json.loads(lines[1][6:])))
        ticket_cid = next((data["conversation_id"] for name, data in ticket_events if name == "session"), None)
        ticket_rows = repository.list_messages(ticket_cid) if ticket_cid else []
        ticket_tools = [data["tool_name"] for name, data in ticket_events if name == "tool_status" and data["state"] == "running"]
        try:
            ticket_data = next((json.loads(row.content) for row in ticket_rows if row.role == "tool"), {})
        except json.JSONDecodeError:
            ticket_data = {}
        with repository.session_factory() as session:
            stored_ticket = session.get(Ticket, ticket_data.get("ticket_id")) if ticket_data.get("ticket_id") else None
        ticket_ok = bool(stored_ticket) and stored_ticket.conversation_id == ticket_cid and ticket_tools == ["create_ticket"] and ticket_events[-1][0] == "done" and [row.role for row in ticket_rows] == ["user", "assistant", "tool", "assistant"] and ticket_rows[1].tool_calls[0]["id"] == ticket_rows[2].tool_call_id
        print(json.dumps({"message": "明确转人工建单", "ticket_id": ticket_data.get("ticket_id"), "pass": ticket_ok}, ensure_ascii=False))
        failures += not ticket_ok
    engine.dispose()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
