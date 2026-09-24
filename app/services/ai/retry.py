"""Exponential backoff + jitter + Retry-After aware async retry."""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from app.services.ai.exceptions import (
    AuthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from app.utils.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")

MAX_DELAY = 20.0


def _backoff(attempt: int) -> float:
    base = min(MAX_DELAY, 2 ** attempt)
    jitter = random.uniform(0, base * 0.3)
    return base + jitter


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    provider: str = "unknown",
) -> T:
    """
    Retries only retryable ProviderErrors.
    Non-retryable errors bubble immediately.
    """
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retries:
        try:
            return await fn()
        except RateLimitError as e:
            last_exc = e
            wait = e.retry_after if e.retry_after is not None else _backoff(attempt)
            wait = max(0.5, min(wait, MAX_DELAY))
            log.warning(
                "retry_rate_limit provider=%s attempt=%s wait=%.2fs",
                provider, attempt + 1, wait,
            )
            await asyncio.sleep(wait)
        except (AuthError, InvalidRequestError, ModelNotFoundError):
            raise
        except ProviderError as e:
            last_exc = e
            if not e.retryable:
                raise
            wait = _backoff(attempt)
            log.warning(
                "retry_provider_error provider=%s attempt=%s wait=%.2fs err=%s",
                provider, attempt + 1, wait, e,
            )
            await asyncio.sleep(wait)
        except asyncio.TimeoutError:
            wait = _backoff(attempt)
            log.warning(
                "retry_timeout provider=%s attempt=%s wait=%.2fs",
                provider, attempt + 1, wait,
            )
            await asyncio.sleep(wait)
            last_exc = ProviderError(provider, "timeout", retryable=True)
        attempt += 1

    assert last_exc is not None
    raise last_exc
