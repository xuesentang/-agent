import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class Settings:
    model_base_url: str
    model_name: str
    model_api_key: str
    structured_output_method: str
    history_token_budget: int
    database_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        values = {}
        for name in ("MODEL_BASE_URL", "MODEL_NAME", "MODEL_API_KEY"):
            value = os.getenv(name, "").strip()
            if not value:
                raise ValueError(f"Missing required setting: {name}")
            values[name] = value
        method = os.getenv("STRUCTURED_OUTPUT_METHOD", "json_mode")
        if method not in {"json_mode", "function_calling"}:
            raise ValueError("STRUCTURED_OUTPUT_METHOD must be json_mode or function_calling")
        try:
            budget = int(os.getenv("HISTORY_TOKEN_BUDGET", "4000"))
        except ValueError as exc:
            raise ValueError("HISTORY_TOKEN_BUDGET must be a positive integer") from exc
        if budget <= 0:
            raise ValueError("HISTORY_TOKEN_BUDGET must be a positive integer")
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("Missing required setting: DATABASE_URL")
        return cls(values["MODEL_BASE_URL"], values["MODEL_NAME"], values["MODEL_API_KEY"], method, budget, database_url)


def build_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.model_base_url,
        model=settings.model_name,
        api_key=settings.model_api_key,
        max_tokens=512,
    )
