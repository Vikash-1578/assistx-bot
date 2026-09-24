"""Lightweight aiohttp health server for Render / Koyeb."""
from __future__ import annotations

from aiohttp import web

from app.utils.logging import get_logger

log = get_logger(__name__)


class HealthServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._ready = False

    async def start(self) -> None:
        self._app = web.Application()
        self._app.router.add_get("/health", self._health)
        self._app.router.add_get("/ready", self._ready_check)
        self._app.router.add_get("/", self._root)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        log.info("health_server_started host=%s port=%s", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        log.info("health_server_stopped")

    def mark_ready(self) -> None:
        self._ready = True

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _ready_check(self, request: web.Request) -> web.Response:
        if self._ready:
            return web.json_response({"status": "ready"})
        return web.json_response({"status": "starting"}, status=503)

    async def _root(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "telegram-ai-agent",
                "endpoints": ["/health", "/ready"],
            }
        )
