"""Application settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_api_key: str = ""
    llm_model: str = "glm-5.3-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    history_token_budget: int = 2048


@lru_cache
def get_settings() -> Settings:
    return Settings()
