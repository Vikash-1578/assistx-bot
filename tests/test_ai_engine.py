"""
AIEngine tests — provider selection, fallback, quota exhaustion.
All providers are MOCKED (no real API calls).
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from app.config import Settings, reset_settings_cache
from app.services.ai.engine import AIEngine
from app.services.ai.exceptions import (
    AllProvidersFailed,
    ProviderError,
    RateLimitError,
)
from app.services.ai.models import AIResponse, ChatMessage, TaskType
from app.services.ai.quota import QuotaTracker


class MockProvider:
    """Mock provider that replays a scripted list of behaviors."""

    def __init__(self, name: str, behaviors: list[Any]) -> None:
        self.name = name
        self.behaviors = behaviors
        self.calls = 0
        self.closed = False

    async def generate(self, messages, *, model, task, temperature=0.7,
                       max_tokens=4096):
        self.calls += 1
        idx = min(self.calls - 1, len(self.behaviors) - 1)
        behavior = self.behaviors[idx]
        if isinstance(behavior, Exception):
            raise behavior
        return AIResponse(
            text=f"{self.name}-reply",
            provider=self.name,
            model=model,
            total_tokens=10,
        )

    async def close(self) -> None:
        self.closed = True


def _mk_settings(monkeypatch, tmp_path, *, groq_key="gsk_x",
                 gemini_key="gmk_x"):
    for key in list(os.environ):
        if any(key.startswith(p) for p in (
            "BOT_TOKEN", "TELEGRAM_", "GROQ_", "GEMINI_",
            "OPENROUTER_", "CEREBRAS_", "SAMBANOVA_", "HF_",
            "AI_", "VECTOR_",
        )):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "123:abc")
    monkeypatch.setenv("AI_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("AI_MAX_RETRIES", "0")  # no sleeps in tests
    monkeypatch.setenv("GROQ_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", groq_key)
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_MODEL_CHAT", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_PRIORITY", "10")
    monkeypatch.setenv("GROQ_DAILY_QUOTA", "100")
    if gemini_key:
        monkeypatch.setenv("GEMINI_ENABLED", "true")
        monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
        monkeypatch.setenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        monkeypatch.setenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash")
        monkeypatch.setenv("GEMINI_PRIORITY", "20")
        monkeypatch.setenv("GEMINI_DAILY_QUOTA", "100")
    reset_settings_cache()
    return Settings()


@pytest.mark.asyncio
async def test_groq_success(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        groq = MockProvider("groq", [None])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        resp = await engine.generate(
            [ChatMessage(role="user", content="hi")],
            task=TaskType.CHAT,
            user_id=1,
        )
        assert resp.text == "groq-reply"
        assert groq.calls == 1
        assert gemini.calls == 0
        snap = await q.snapshot("groq", 100)
        assert snap["requests"] == 1
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_fallback_after_rate_limit(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        groq = MockProvider("groq", [RateLimitError("groq", "429")])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        resp = await engine.generate(
            [ChatMessage(role="user", content="hi")],
            task=TaskType.CHAT,
            user_id=1,
        )
        assert resp.text == "gemini-reply"
        assert groq.calls == 1
        assert gemini.calls == 1
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_fallback_after_provider_error(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        groq = MockProvider("groq", [ProviderError("groq", "boom", retryable=False)])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        resp = await engine.generate(
            [ChatMessage(role="user", content="hi")],
            task=TaskType.CHAT,
            user_id=1,
        )
        assert resp.text == "gemini-reply"
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_all_providers_fail(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        groq = MockProvider("groq", [ProviderError("groq", "boom", retryable=False)])
        gemini = MockProvider("gemini", [ProviderError("gemini", "boom", retryable=False)])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        with pytest.raises(AllProvidersFailed):
            await engine.generate(
                [ChatMessage(role="user", content="hi")],
                task=TaskType.CHAT,
                user_id=1,
            )
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_cooldown_after_failures(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        # groq always fails; gemini always works
        groq = MockProvider("groq", [ProviderError("groq", "x", retryable=False)])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        # 3 calls to trigger cooldown (threshold = 3)
        for _ in range(3):
            await engine.generate(
                [ChatMessage(role="user", content="hi")],
                task=TaskType.CHAT,
                user_id=1,
            )
        # next call: groq should be in cooldown → gemini only
        groq_calls_before = groq.calls
        await engine.generate(
            [ChatMessage(role="user", content="hi")],
            task=TaskType.CHAT,
            user_id=1,
        )
        assert groq.calls == groq_calls_before, "groq should be in cooldown"
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_quota_exhaustion_skips_provider(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    # set groq daily quota to 1
    monkeypatch.setenv("GROQ_DAILY_QUOTA", "1")
    reset_settings_cache()
    s = Settings()
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        # fill groq quota
        await q.record_success("groq")
        groq = MockProvider("groq", [None])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        resp = await engine.generate(
            [ChatMessage(role="user", content="hi")],
            task=TaskType.CHAT,
            user_id=1,
        )
        assert resp.text == "gemini-reply"
        assert groq.calls == 0
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_status_endpoint(monkeypatch, tmp_path):
    s = _mk_settings(monkeypatch, tmp_path)
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        groq = MockProvider("groq", [None])
        gemini = MockProvider("gemini", [None])
        engine = AIEngine(s, {"groq": groq, "gemini": gemini}, q)  # type: ignore[arg-type]
        statuses = await engine.status()
        names = {st.name for st in statuses}
        assert names == {"groq", "gemini"}
        for st in statuses:
            assert st.enabled is True
            assert st.available is True
    finally:
        await q.close()
