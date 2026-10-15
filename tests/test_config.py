from app.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://open.bigmodel.cn/api/paas/v4/"
    assert settings.llm_model == "glm-5.3-flash"
    assert settings.llm_api_key == ""
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 512
    assert settings.history_token_budget == 2048


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "4096")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "deepseek-chat"
    assert settings.history_token_budget == 4096
    assert settings.llm_temperature == 0.2
