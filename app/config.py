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
    llm_timeout: int = 60
    llm_max_retries: int = 1
    # with_structured_output method: "json_mode" (GLM) or "function_calling".
    llm_structured_method: str = "json_mode"
    history_token_budget: int = 2048
    # CH02: database (Docker MySQL on host port 3307, see docker-compose.yml).
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent"
    db_echo: bool = False
    # CH02: tool execution timeout (registry executor).
    tool_timeout_seconds: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
