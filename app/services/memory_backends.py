"""
Where chat memory is stored so it survives restarts.

  SqliteBackend   - a local file (great on a VPS / your PC; NOT on Heroku,
                    whose disk is wiped on every restart)
  SupabaseBackend - Supabase (Postgres) over its REST API (use on Heroku)

Standard library only: SQLite via sqlite3 and HTTP via urllib, both run in
worker threads so the bot's event loop never blocks.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

log = get_logger(__name__)


class MemoryBackendError(RuntimeError):
    """Storage failed (network, HTTP error, bad response)."""


@dataclass
class UserMemory:
    messages: list[tuple[str, str]] = field(default_factory=list)  # oldest first
    summary: str = ""
    facts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- SQLite
_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistx_messages (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role    TEXT    NOT NULL,
    content TEXT    NOT NULL,
    ts      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assistx_messages_user
    ON assistx_messages(user_id, id);
CREATE TABLE IF NOT EXISTS assistx_profiles (
    user_id INTEGER PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    facts   TEXT NOT NULL DEFAULT '[]'
);
"""


class SqliteBackend:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _open(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _run(self, fn):
        with self._lock:
            assert self._conn is not None, "SqliteBackend.init() not called"
            try:
                return fn(self._conn)
            finally:
                self._conn.commit()

    async def init(self) -> None:
        await asyncio.to_thread(self._open)
        log.info("memory_sqlite_ready path=%s", self.path)

    async def close(self) -> None:
        def _close():
            with self._lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None

        await asyncio.to_thread(_close)

    async def load(self, user_id: int, limit: int) -> UserMemory:
        def _load(c: sqlite3.Connection) -> UserMemory:
            rows = c.execute(
                "SELECT role, content FROM assistx_messages WHERE user_id=? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            prof = c.execute(
                "SELECT summary, facts FROM assistx_profiles WHERE user_id=?",
                (user_id,),
            ).fetchone()
            facts: list[str] = []
            if prof:
                try:
                    facts = [str(f) for f in json.loads(prof[1] or "[]")]
                except json.JSONDecodeError:
                    facts = []
            return UserMemory(
                messages=[(r[0], r[1]) for r in reversed(rows)],
                summary=(prof[0] if prof else "") or "",
                facts=facts,
            )

        return await asyncio.to_thread(self._run, _load)

    async def append(self, user_id: int, role: str, content: str) -> None:
        await asyncio.to_thread(
            self._run,
            lambda c: c.execute(
                "INSERT INTO assistx_messages(user_id, role, content, ts) "
                "VALUES (?,?,?,?)",
                (user_id, role, content, time.time()),
            ),
        )

    async def clear_history(self, user_id: int) -> None:
        def _clear(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM assistx_messages WHERE user_id=?", (user_id,))
            c.execute(
                "INSERT INTO assistx_profiles(user_id, summary) VALUES (?, '') "
                "ON CONFLICT(user_id) DO UPDATE SET summary=''",
                (user_id,),
            )

        await asyncio.to_thread(self._run, _clear)

    async def trim(self, user_id: int, keep: int) -> None:
        await asyncio.to_thread(
            self._run,
            lambda c: c.execute(
                "DELETE FROM assistx_messages WHERE user_id=? AND id <= ("
                "SELECT id FROM assistx_messages WHERE user_id=? "
                "ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (user_id, user_id, keep),
            ),
        )

    async def set_summary(self, user_id: int, summary: str) -> None:
        await asyncio.to_thread(
            self._run,
            lambda c: c.execute(
                "INSERT INTO assistx_profiles(user_id, summary) VALUES (?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary",
                (user_id, summary),
            ),
        )

    async def set_facts(self, user_id: int, facts: list[str]) -> None:
        await asyncio.to_thread(
            self._run,
            lambda c: c.execute(
                "INSERT INTO assistx_profiles(user_id, facts) VALUES (?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET facts=excluded.facts",
                (user_id, json.dumps(facts, ensure_ascii=False)),
            ),
        )


# -------------------------------------------------------------- Supabase
class SupabaseBackend:
    """PostgREST client. Tables: see docs/supabase_schema.sql."""

    def __init__(self, url: str, key: str, timeout: float = 15.0) -> None:
        if not url or not key:
            raise MemoryBackendError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self.base = url.rstrip("/") + "/rest/v1"
        self.key = key
        self.timeout = timeout

    def _headers(self, prefer: str | None) -> dict[str, str]:
        h = {
            "apikey": self.key,
            "Content-Type": "application/json",
            "User-Agent": "assistx-bot/1.0",
        }
        if self.key.startswith("eyJ"):  # legacy JWT-style keys also go in Authorization
            h["Authorization"] = f"Bearer {self.key}"
        if prefer:
            h["Prefer"] = prefer
        return h

    def _call(
        self,
        method: str,
        table: str,
        query: dict[str, str] | None = None,
        body: Any = None,
        prefer: str | None = None,
    ) -> Any:
        url = f"{self.base}/{table}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, headers=self._headers(prefer), method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            raise MemoryBackendError(f"supabase {method} {table}: http {e.code} {detail}")
        except Exception as e:
            raise MemoryBackendError(f"supabase {method} {table}: {type(e).__name__}: {e}")
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise MemoryBackendError(f"supabase {method} {table}: bad JSON: {e}")

    async def _acall(self, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self._call, *args, **kwargs)

    async def init(self) -> None:
        # Cheap connectivity/schema check: fails fast if the tables are missing.
        await self._acall("GET", "assistx_profiles", {"select": "user_id", "limit": "1"})
        log.info("memory_supabase_ready")

    async def close(self) -> None:
        return None

    async def load(self, user_id: int, limit: int) -> UserMemory:
        rows = await self._acall(
            "GET",
            "assistx_messages",
            {
                "user_id": f"eq.{user_id}",
                "select": "role,content",
                "order": "id.desc",
                "limit": str(limit),
            },
        )
        prof = await self._acall(
            "GET",
            "assistx_profiles",
            {"user_id": f"eq.{user_id}", "select": "summary,facts"},
        )
        p = prof[0] if prof else {}
        facts = p.get("facts") or []
        return UserMemory(
            messages=[(r["role"], r["content"]) for r in reversed(rows or [])],
            summary=p.get("summary") or "",
            facts=[str(f) for f in facts] if isinstance(facts, list) else [],
        )

    async def append(self, user_id: int, role: str, content: str) -> None:
        await self._acall(
            "POST",
            "assistx_messages",
            body=[{"user_id": user_id, "role": role, "content": content}],
            prefer="return=minimal",
        )

    async def _upsert_profile(self, user_id: int, **cols: Any) -> None:
        await self._acall(
            "POST",
            "assistx_profiles",
            {"on_conflict": "user_id"},
            body={"user_id": user_id, **cols},
            prefer="resolution=merge-duplicates,return=minimal",
        )

    async def clear_history(self, user_id: int) -> None:
        await self._acall(
            "DELETE", "assistx_messages", {"user_id": f"eq.{user_id}"},
            prefer="return=minimal",
        )
        await self._upsert_profile(user_id, summary="")

    async def trim(self, user_id: int, keep: int) -> None:
        rows = await self._acall(
            "GET",
            "assistx_messages",
            {
                "user_id": f"eq.{user_id}",
                "select": "id",
                "order": "id.desc",
                "offset": str(keep),
                "limit": "1",
            },
        )
        if rows:
            await self._acall(
                "DELETE",
                "assistx_messages",
                {"user_id": f"eq.{user_id}", "id": f"lte.{rows[0]['id']}"},
                prefer="return=minimal",
            )

    async def set_summary(self, user_id: int, summary: str) -> None:
        await self._upsert_profile(user_id, summary=summary)

    async def set_facts(self, user_id: int, facts: list[str]) -> None:
        await self._upsert_profile(user_id, facts=facts)


def build_memory_backend(cfg) -> SqliteBackend | SupabaseBackend | None:
    """Create the backend named in settings.memory (None = RAM only)."""
    if cfg.backend == "ram":
        return None
    if cfg.backend == "supabase":
        return SupabaseBackend(cfg.supabase_url, cfg.supabase_key)
    return SqliteBackend(cfg.db_path)
