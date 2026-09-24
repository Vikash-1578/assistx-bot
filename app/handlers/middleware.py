"""
Outer middleware:
- Authorization (allowlist)
- Per-user rate limiting
- Passes dependencies (settings, router, memory) into handler data
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from app.services.agent_router import AgentRouter
from app.services.memory import MemoryStore
from app.utils.logging import get_logger
from app.utils.security import AllowList, RateLimiter

log = get_logger(__name__)


class AuthAndRateLimitMiddleware(BaseMiddleware):
    def __init__(
        self,
        allowlist: AllowList,
        rate_limiter: RateLimiter,
    ) -> None:
        self.allowlist = allowlist
        self.rate_limiter = rate_limiter

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        # --- Authorization ---
        if not self.allowlist.is_allowed(user.id):
            log.warning("unauthorized_user user_id=%s", user.id)
            if isinstance(event, Message):
                try:
                    await event.answer(
                        "⛔ You are not authorized to use this bot."
                    )
                except Exception:
                    pass
            return None

        # --- Rate limit ---
        allowed = await self.rate_limiter.allow(user.id)
        if not allowed:
            retry = await self.rate_limiter.retry_after(user.id)
            log.warning("rate_limited user_id=%s retry_after=%ss", user.id, retry)
            if isinstance(event, Message):
                try:
                    await event.answer(
                        f"⏳ Too many requests. Please try again in {retry}s."
                    )
                except Exception:
                    pass
            return None

        return await handler(event, data)


class DependencyMiddleware(BaseMiddleware):
    """Injects shared services into handler `data`."""

    def __init__(
        self,
        agent_router: AgentRouter,
        memory: MemoryStore,
    ) -> None:
        self.agent_router = agent_router
        self.memory = memory

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["agent_router"] = self.agent_router
        data["memory"] = self.memory
        return await handler(event, data)
