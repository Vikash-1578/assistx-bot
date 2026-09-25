"""
Application entry point.

Lifecycle:
  validate config → logging → quota db → providers → engine
  → bot → dispatcher → middlewares → routers → health server
  → polling → (SIGINT/SIGTERM) → graceful shutdown
"""
from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.config import Settings, get_settings
from app.handlers import chat as chat_h
from app.handlers import code as code_h
from app.handlers import content as content_h
from app.handlers import errors as errors_h
from app.handlers import files as files_h
from app.handlers import images as images_h
from app.handlers import freelance as freelance_h
from app.handlers import start as start_h
from app.handlers import summary as summary_h
from app.handlers.middleware import (
    AuthAndRateLimitMiddleware,
    DependencyMiddleware,
)
from app.services.agent_router import AgentRouter
from app.services.ai.engine import AIEngine
from app.services.ai.factory import build_providers
from app.services.ai.quota import QuotaTracker
from app.services.health_server import HealthServer
from app.services.memory import MemoryStore
from app.utils.logging import configure_logging, get_logger
from app.utils.security import AllowList, RateLimiter

log = get_logger(__name__)


class Application:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None
        self.engine: AIEngine | None = None
        self.router: AgentRouter | None = None
        self.memory = MemoryStore()
        self.quota: QuotaTracker | None = None
        self.health: HealthServer | None = None
        self._shutdown = asyncio.Event()

    async def setup(self) -> None:
        s = self.settings

        # --- Quota DB ---
        self.quota = QuotaTracker(s.ai.quota_db_path)
        await self.quota.init()

        # --- Providers + Engine ---
        providers = build_providers(s)
        if not providers:
            raise RuntimeError("No providers built from settings.")
        self.engine = AIEngine(s, providers, self.quota)
        self.router = AgentRouter(self.engine)

        # --- Bot + Dispatcher ---
        self.bot = Bot(
            token=s.telegram.bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.dp = Dispatcher(storage=MemoryStorage())

        # --- Middlewares (outer) ---
        allowlist = AllowList(s.telegram.allowed_user_ids)
        if not s.telegram.allowed_user_ids:
            log.warning(
                "allowlist_empty_all_users_allowed "
                "(set TELEGRAM_ALLOWED_USER_IDS for production)"
            )
        rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        self.dp.update.outer_middleware(
            AuthAndRateLimitMiddleware(allowlist, rate_limiter)
        )
        self.dp.update.outer_middleware(
            DependencyMiddleware(self.router, self.memory)
        )

        # --- Routers ---
        self.dp.include_router(start_h.router)
        self.dp.include_router(freelance_h.router)
        self.dp.include_router(code_h.router)
        self.dp.include_router(content_h.router)
        self.dp.include_router(summary_h.router)
        self.dp.include_router(files_h.router)
        self.dp.include_router(images_h.router)
        self.dp.include_router(chat_h.router)  # last: catch-all text
        self.dp.include_router(errors_h.router)

        # --- Health server ---
        port = int(os.environ.get("PORT", s.health.port))
        self.health = HealthServer(s.health.host, port)

        log.info("setup_complete providers=%s", list(providers.keys()))

    async def run(self) -> None:
        assert self.dp is not None
        assert self.bot is not None
        assert self.health is not None

        await self.health.start()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown.set)
            except NotImplementedError:
                # Windows fallback
                signal.signal(sig, lambda *_: self._shutdown.set())

        me = await self.bot.get_me()
        log.info("bot_online username=%s id=%s", me.username, me.id)

        # Ensure polling can start: remove any stale webhook
        try:
            await self.bot.delete_webhook(drop_pending_updates=True)
            log.info("webhook_cleared")
        except Exception as e:
            log.warning("delete_webhook_failed err=%s", e)

        self.health.mark_ready()

        polling_task = asyncio.create_task(
            self.dp.start_polling(
                self.bot,
                allowed_updates=self.dp.resolve_used_update_types(),
            )
        )
        shutdown_task = asyncio.create_task(self._shutdown.wait())

        done, pending = await asyncio.wait(
            {polling_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if polling_task in done and polling_task.exception():
            log.error("polling_crashed err=%s", polling_task.exception())

    async def shutdown(self) -> None:
        log.info("shutdown_started")
        try:
            if self.dp is not None:
                await self.dp.stop_polling()
        except Exception as e:
            log.warning("stop_polling_error err=%s", e)

        try:
            if self.bot is not None:
                await self.bot.session.close()
        except Exception as e:
            log.warning("bot_close_error err=%s", e)

        if self.engine is not None:
            await self.engine.close()
        if self.quota is not None:
            await self.quota.close()
        if self.health is not None:
            await self.health.stop()

        log.info("shutdown_complete")


async def _async_main() -> int:
    try:
        settings = get_settings()
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        return 2

    configure_logging(settings.logging)

    app = Application(settings)
    try:
        await app.setup()
    except Exception as e:
        log.exception("setup_failed err=%s", e)
        return 3

    try:
        await app.run()
    except Exception as e:
        log.exception("runtime_failed err=%s", e)
    finally:
        await app.shutdown()
    return 0


def main() -> None:
    try:
        rc = asyncio.run(_async_main())
    except KeyboardInterrupt:
        rc = 0
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
