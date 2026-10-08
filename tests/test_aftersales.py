import asyncio
import json
from pathlib import Path

import httpx
import pytest

from customer_service.aftersales import AfterSalesInfo, ExtractionService
from customer_service.chat import ChatService
from customer_service.history import ConversationStore
from customer_service.main import create_app


class FakeStructured:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, messages):
        assert "JSON" in messages[0].content
        return self.result


class FakeModel:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, method):
        assert schema is AfterSalesInfo
        assert method == "json_mode"
        return FakeStructured(self.result)


def test_labeled_examples():
    async def run():
        path = Path(__file__).parents[1] / "evals" / "aftersales.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            example = json.loads(line)
            service = ExtractionService(FakeModel(example["expected"]), "json_mode")
            assert (await service.extract(example["description"])).model_dump() == example["expected"]
    asyncio.run(run())


def test_invalid_category_and_empty_response_fail():
    async def run():
        for result in ({"order_id": None, "request_type": "wrong", "expected_resolution": None}, None):
            service = ExtractionService(FakeModel(result), "json_mode")
            with pytest.raises(ValueError):
                await service.extract("描述")
    asyncio.run(run())


def test_extract_route_and_empty_input():
    async def run():
        expected = {"order_id": "A123", "request_type": "refund", "expected_resolution": "退款"}
        service = ExtractionService(FakeModel(expected), "json_mode")
        app = create_app(ChatService(None, ConversationStore(), 4000), service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/aftersales/extract", json={"description": "订单 A123，请退款"})
            assert response.json() == expected
            assert (await client.post("/aftersales/extract", json={"description": " "})).status_code == 422
    asyncio.run(run())
