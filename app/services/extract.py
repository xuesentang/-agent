"""After-sales structured extraction service (spec D2)."""

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.core.llm import get_extract_model
from app.core.prompts import build_extract_prompt
from app.core.schemas import AfterSalesInfo


class ExtractionError(Exception):
    """Raised when setup, upstream call, or schema validation fails."""


async def extract_after_sales(text: str, model: ChatOpenAI | None = None) -> dict:
    base_model = model if model is not None else get_extract_model()
    try:
        structured = base_model.with_structured_output(
            AfterSalesInfo, method="function_calling"
        )
    except Exception as exc:
        raise ExtractionError(f"structured output setup failed: {exc}") from exc

    messages = build_extract_prompt().format_messages(text=text)
    try:
        result = await structured.ainvoke(messages)
    except Exception as exc:
        raise ExtractionError(f"extraction call failed: {exc}") from exc

    if not isinstance(result, AfterSalesInfo):
        # Some OpenAI-compatible providers hand back a raw dict; validate manually.
        try:
            result = AfterSalesInfo.model_validate(result)
        except (ValidationError, TypeError) as exc:
            raise ExtractionError(f"model output does not match schema: {exc}") from exc
    return result.model_dump()
