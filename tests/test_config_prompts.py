import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.config import Settings, build_model
from customer_service.prompts import CHAT_PROMPT, EXTRACTION_PROMPT


def test_settings_load_and_model(monkeypatch):
    for key, value in {
        "MODEL_BASE_URL": "https://example.test/v1",
        "MODEL_NAME": "test-model",
        "MODEL_API_KEY": "secret-value",
        "STRUCTURED_OUTPUT_METHOD": "json_mode",
        "HISTORY_TOKEN_BUDGET": "2000",
        "DATABASE_URL": "sqlite:///test.db",
    }.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.model_name == "test-model"
    assert settings.history_token_budget == 2000
    model = build_model(settings)
    assert model.model_name == "test-model"
    assert model.max_tokens == 512


def test_missing_key_is_rejected_without_secret(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MODEL_API_KEY", "")
    with pytest.raises(ValueError, match="MODEL_API_KEY") as exc:
        Settings.from_env()
    assert "secret-value" not in str(exc.value)


def test_prompts_have_roles_and_json_instruction():
    messages = CHAT_PROMPT.format_messages(history=[], message="你好")
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert "JSON" in EXTRACTION_PROMPT.format_messages(description="退货")[0].content
