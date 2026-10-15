import pytest

from app.core.schemas import AfterSalesInfo
from app.services.extract import ExtractionError, extract_after_sales


class FakeStructuredModel:
    """Stands in for with_structured_output(...) result."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


class FakeModel:
    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, schema, method=None):
        assert method == "function_calling"
        assert schema is AfterSalesInfo
        return self._structured


async def test_extract_returns_field_dict():
    info = AfterSalesInfo(
        order_id="123", claim_type="refund", expected_solution="全额退款"
    )
    model = FakeModel(FakeStructuredModel(info))
    result = await extract_after_sales("订单123鞋子开胶要退款", model)
    assert result == {
        "order_id": "123",
        "claim_type": "refund",
        "expected_solution": "全额退款",
    }


async def test_extract_accepts_raw_dict_from_model():
    # 有些供应商经 with_structured_output 仍返回裸 dict，需手动校验兜底
    raw = {"order_id": None, "claim_type": "inquiry", "expected_solution": None}
    model = FakeModel(FakeStructuredModel(raw))
    result = await extract_after_sales("想问问退货政策", model)
    assert result["claim_type"] == "inquiry"


async def test_extract_invalid_output_raises():
    model = FakeModel(FakeStructuredModel({"claim_type": "not-a-type"}))
    with pytest.raises(ExtractionError):
        await extract_after_sales("随便说点啥", model)


async def test_extract_upstream_error_wrapped():
    class BrokenModel:
        def with_structured_output(self, schema, method=None):
            raise RuntimeError("connection refused")

    with pytest.raises(ExtractionError):
        await extract_after_sales("文本", BrokenModel())
