from pathlib import Path

from app.config import get_settings
from app.core.llm import get_chat_model, get_extract_model


def _reset(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "777")
    get_settings.cache_clear()


def test_chat_model_uses_settings(monkeypatch):
    _reset(monkeypatch)
    model = get_chat_model()
    assert model.model_name == "test-model"
    assert model.temperature == 0.7
    assert model.max_tokens == 777


def test_extract_model_temperature_zero(monkeypatch):
    _reset(monkeypatch)
    assert get_extract_model().temperature == 0


def test_no_hardcoded_key():
    app_root = Path(__file__).resolve().parents[1] / "app"
    for path in app_root.rglob("*.py"):
        assert "sk-" not in path.read_text(encoding="utf-8"), path
