"""Config tests — validation, defaults, parsing."""
from __future__ import annotations

import os

import pytest

from app.config import Settings, TelegramSettings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "BOT_TOKEN",
                "TELEGRAM_",
                "GROQ_",
                "GEMINI_",
                "OPENROUTER_",
                "CEREBRAS_",
                "SAMBANOVA_",
                "HF_",
                "AI_",
                "VECTOR_",
                "HEALTH_",
                "LOG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def _base_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123456:ABCdef")
    monkeypatch.setenv("GROQ_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_MODEL_CHAT", "llama-3.3-70b-versatile")


def test_valid_config(monkeypatch, tmp_path):
    _base_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # avoid reading real .env
    s = Settings()
    assert s.telegram.bot_token == "123456:ABCdef"
    assert s.groq.enabled is True
    assert s.groq.api_key == "gsk_test"


def test_missing_bot_token(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    with pytest.raises(Exception):
        Settings()


def test_no_providers_configured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "123:abc")
    with pytest.raises(Exception):
        Settings()


def test_enabled_provider_without_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "123:abc")
    monkeypatch.setenv("GROQ_ENABLED", "true")
    # missing GROQ_API_KEY
    with pytest.raises(Exception):
        Settings()


def test_invalid_integer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _base_env(monkeypatch)
    monkeypatch.setenv("AI_REQUEST_TIMEOUT", "not-a-number")
    with pytest.raises(Exception):
        Settings()


def test_allowed_user_ids_parsing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _base_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "111,222, 333")
    s = Settings()
    assert s.telegram.allowed_user_ids == [111, 222, 333]


def test_bot_token_validation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "malformed")
    monkeypatch.setenv("GROQ_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    with pytest.raises(Exception):
        Settings()
