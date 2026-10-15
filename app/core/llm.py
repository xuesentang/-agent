"""ChatOpenAI factory driven entirely by settings (spec D1)."""

from langchain_openai import ChatOpenAI

from app.config import get_settings


def _base_model(temperature: float) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,  # review Minor #7: never hang SSE on a silent upstream
        max_retries=settings.llm_max_retries,
    )


def get_chat_model() -> ChatOpenAI:
    """Model for the customer-service conversation (temperature from settings)."""
    return _base_model(get_settings().llm_temperature)


def get_extract_model() -> ChatOpenAI:
    """Model for structured extraction (temperature 0 for determinism)."""
    return _base_model(0)
