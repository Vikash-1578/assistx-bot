"""
Security primitives:

- Allowlist check (numeric Telegram user ID)
- Async-safe per-user rate limiter
- Input sanitization helpers
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.utils.logging import get_logger

log = get_logger(__name__)


class AllowList:
    def __init__(self, allowed_ids: list[int]) -> None:
        self._allowed = set(allowed_ids)

    def is_allowed(self, user_id: int) -> bool:
        if not self._allowed:
            return True
        return user_id in self._allowed

    def __contains__(self, user_id: int) -> bool:
        return self.is_allowed(user_id)


@dataclass
class _Bucket:
    hits: deque[float] = field(default_factory=deque)


class RateLimiter:
    """
    Sliding-window per-user rate limiter.

    Default: 20 requests / 60 seconds per user.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[int, _Bucket] = defaultdict(_Bucket)
        self._lock = asyncio.Lock()

    async def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets[user_id]
            cutoff = now - self.window_seconds
            while bucket.hits and bucket.hits[0] < cutoff:
                bucket.hits.popleft()
            if len(bucket.hits) >= self.max_requests:
                return False
            bucket.hits.append(now)
            return True

    async def retry_after(self, user_id: int) -> int:
        async with self._lock:
            bucket = self._buckets.get(user_id)
            if not bucket or not bucket.hits:
                return 0
            elapsed = time.monotonic() - bucket.hits[0]
            remaining = self.window_seconds - elapsed
            return max(0, int(remaining) + 1)


MAX_USER_MESSAGE_CHARS = 8_000


def sanitize_user_text(text: str | None) -> str:
    """Trim + bound user input length. Never raises."""
    if not text:
        return ""
    text = text.strip()
    if len(text) > MAX_USER_MESSAGE_CHARS:
        text = text[:MAX_USER_MESSAGE_CHARS]
    return text


def is_authorized(user_id: int, allowlist: AllowList) -> bool:
    ok = allowlist.is_allowed(user_id)
    if not ok:
        log.warning("unauthorized_access_attempt user_id=%s", user_id)
    return ok
