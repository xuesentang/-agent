"""Exercise the three chapter acceptance criteria against the configured upstream."""

import asyncio
import json

import httpx

from customer_service.main import _build_app


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith("event: ") and lines[1].startswith("data: "):
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


async def main() -> int:
    app = _build_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", timeout=120) as client:
        first = await client.post("/chat/stream", json={"message": "我的订单编号是 TEST-A123。请记住，简短回复。"})
        first_events = parse_events(first.text)
        first_tokens = [data["text"] for name, data in first_events if name == "token"]
        first_reply = "".join(first_tokens)
        asks_for_identity = any(term in first_reply for term in ("手机号", "收件人姓名", "身份证", "地址"))
        first_ok = first.status_code == 200 and len(first_tokens) >= 2 and first_events[-1][0] == "done" and not asks_for_identity
        print(json.dumps({"step": "stream", "ok": first_ok, "status": first.status_code, "events": [name for name, _ in first_events], "reply": first_reply}, ensure_ascii=False))
        if not first_ok:
            return 1

        conversation_id = first_events[0][1]["conversation_id"]
        second = await client.post("/chat/stream", json={"conversation_id": conversation_id, "message": "刚才我说的订单编号是什么？只回答编号。"})
        second_events = parse_events(second.text)
        second_reply = "".join(data["text"] for name, data in second_events if name == "token")
        second_ok = second.status_code == 200 and second_events[-1][0] == "done" and "TEST-A123" in second_reply
        print(json.dumps({"step": "memory", "ok": second_ok, "status": second.status_code, "reply": second_reply}, ensure_ascii=False))

        extraction = await client.post("/aftersales/extract", json={"description": "订单 TEST-A123 的耳机坏了，我想退款。"})
        result = extraction.json()
        extract_ok = extraction.status_code == 200 and result.get("order_id") == "TEST-A123" and result.get("request_type") == "refund" and bool(result.get("expected_resolution"))
        print(json.dumps({"step": "extraction", "ok": extract_ok, "status": extraction.status_code, "result": result}, ensure_ascii=False))
        return 0 if second_ok and extract_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
