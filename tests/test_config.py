from app.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-flash"
    assert settings.llm_api_key == ""
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 512
    assert settings.history_token_budget == 2048


def test_settings_thinking_defaults_to_disabled():
    """思考模式必须默认关:开着的时候长输出 9398 token / 36 秒,关了 764 token / 6 秒。"""
    settings = Settings(_env_file=None)
    assert settings.llm_thinking == "disabled"
    assert settings.recall_route_limit == 20
    assert settings.judge_model == ""


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "4096")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "deepseek-chat"
    assert settings.history_token_budget == 4096
    assert settings.llm_temperature == 0.2


def test_langfuse_host_accepts_base_url_alias(monkeypatch):
    """美区账号按 SDK 文档把 host 写成 LANGFUSE_BASE_URL,两个名字都得认。"""
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    assert Settings(_env_file=None).langfuse_host == "https://cloud.langfuse.com"

    monkeypatch.delenv("LANGFUSE_HOST")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    assert Settings(_env_file=None).langfuse_host == "https://us.cloud.langfuse.com"
