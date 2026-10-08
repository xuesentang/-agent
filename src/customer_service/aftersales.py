from typing import Literal

from pydantic import BaseModel

from customer_service.prompts import EXTRACTION_PROMPT


class AfterSalesInfo(BaseModel):
    order_id: str | None
    request_type: Literal["refund", "return", "exchange", "repair", "logistics", "other"]
    expected_resolution: str | None


class ExtractionService:
    def __init__(self, model, method: str) -> None:
        self.model = model.with_structured_output(AfterSalesInfo, method=method)

    async def extract(self, description: str) -> AfterSalesInfo:
        messages = EXTRACTION_PROMPT.format_messages(description=description)
        result = await self.model.ainvoke(messages)
        if result is None:
            raise ValueError("Structured model returned no content")
        if isinstance(result, AfterSalesInfo):
            return result
        return AfterSalesInfo.model_validate(result)
