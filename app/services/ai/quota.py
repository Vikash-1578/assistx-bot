"""SQLite-backed daily quota tracker (UTC)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_usage (
    provider      TEXT NOT NULL,
    usage_date    TEXT NOT NULL,
    requests      INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    failures      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, usage_date)
);
CREATE INDEX IF NOT EXISTS idx_provider_usage_date ON provider_usage(usage_date);
"""


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class QuotaTracker:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("quota_db_ready path=%s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.init()
        assert self._db is not None
        return self._db

    async def _row(self, provider: str) -> tuple[int, int, int, int, int]:
        db = await self._ensure()
        async with db.execute(
            "SELECT requests, input_tokens, output_tokens, total_tokens, failures "
            "FROM provider_usage WHERE provider=? AND usage_date=?",
            (provider, _today_utc()),
        ) as cur:
            row = await cur.fetchone()
            return row if row else (0, 0, 0, 0, 0)

    async def requests_today(self, provider: str) -> int:
        return (await self._row(provider))[0]

    async def failures_today(self, provider: str) -> int:
        return (await self._row(provider))[4]

    async def record_success(
        self,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        db = await self._ensure()
        today = _today_utc()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO provider_usage
                    (provider, usage_date, requests, input_tokens, output_tokens, total_tokens, failures)
                VALUES (?, ?, 1, ?, ?, ?, 0)
                ON CONFLICT(provider, usage_date) DO UPDATE SET
                    requests      = requests + 1,
                    input_tokens  = input_tokens + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    total_tokens  = total_tokens + excluded.total_tokens
                """,
                (provider, today, input_tokens, output_tokens, total_tokens),
            )
            await db.commit()

    async def record_failure(self, provider: str) -> None:
        db = await self._ensure()
        today = _today_utc()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO provider_usage
                    (provider, usage_date, requests, failures)
                VALUES (?, ?, 0, 1)
                ON CONFLICT(provider, usage_date) DO UPDATE SET
                    failures = failures + 1
                """,
                (provider, today),
            )
            await db.commit()

    async def snapshot(self, provider: str, daily_limit: int) -> dict:
        req, itok, otok, ttok, fail = await self._row(provider)
        return {
            "provider": provider,
            "date": _today_utc(),
            "requests": req,
            "input_tokens": itok,
            "output_tokens": otok,
            "total_tokens": ttok,
            "failures": fail,
            "daily_limit": daily_limit,
            "remaining": max(0, daily_limit - req),
        }
