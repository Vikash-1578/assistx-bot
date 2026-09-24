"""Retry logic tests."""
from __future__ import annotations

import pytest

from app.services.ai.exceptions import (
    AuthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from app.services.ai.retry import with_retry


@pytest.mark.asyncio
async def test_success_first_try():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        return "ok"

    result = await with_retry(fn, max_retries=2, provider="test")
    assert result == "ok"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_on_retryable_then_success():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("test", "temp", retryable=True)
        return "ok"

    result = await with_retry(fn, max_retries=2, provider="test")
    assert result == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_no_retry_on_auth_error():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise AuthError("test", "bad key")

    with pytest.raises(AuthError):
        await with_retry(fn, max_retries=2, provider="test")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_no_retry_on_invalid_request():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise InvalidRequestError("test", "bad payload")

    with pytest.raises(InvalidRequestError):
        await with_retry(fn, max_retries=2, provider="test")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_rate_limit_with_retry_after(monkeypatch):
    # Avoid real sleeps
    import app.services.ai.retry as r

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(r.asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError("test", "429", retry_after=1.0)
        return "ok"

    result = await with_retry(fn, max_retries=2, provider="test")
    assert result == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_exhausted_retries_raises(monkeypatch):
    import app.services.ai.retry as r

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(r.asyncio, "sleep", fake_sleep)

    async def fn():
        raise ProviderError("test", "always fails", retryable=True)

    with pytest.raises(ProviderError):
        await with_retry(fn, max_retries=2, provider="test")
