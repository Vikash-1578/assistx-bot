"""
Chat memory.

* Short-term: the last N messages per user, kept in RAM (fast, sync API that
  every handler already uses: get / add_user / add_assistant / clear / size).
* Persistent: every message is also written to a backend (SQLite or Supabase)
  by a background queue, and loaded again when the bot restarts.
* Rolling summary: when the chat grows past N messages, the oldest ones are
  folded into a short summary by the AI (via `set_summarizer`), so long
  conversations are not forgotten.
* Facts: things the user asked the bot to remember ("yaad rakhna ...",
  /remember). Survive /reset.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable

from app.services.ai.models import ChatMessage
from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_HISTORY_MESSAGES = 30
MAX_FACTS = 50
MAX_FACT_CHARS = 300
MAX_OVERFLOW = 200

Summarizer = Callable[[str, list[ChatMessage]], Awaitable[str]]

_REMEMBER = re.compile(
    r"^\s*(?:please\s+|plz\s+)?(?:yaad\s+rakh(?:na|o|iye|ana)?|yaad\s+rkh(?:na|o)?|"
    r"remember(?:\s+that)?|note\s+(?:this|that)\s+down|note\s+that)"
    r"\b\s*[:,\-]?\s*(.+)$",
    re.I | re.S,
)
_FACT_LEAD = re.compile(r"^(ki|ke|ye|yeh|that|this)\s+", re.I)
_NOT_A_FACT = re.compile(r"^(when|how|what|why|kya|kaise|kab|kyun|where)\b", re.I)


def extract_fact_request(text: str) -> str | None:
    """'yaad rakhna ki mera naam Vikash hai' -> 'mera naam Vikash hai'."""
    m = _REMEMBER.match(text or "")
    if not m:
        return None
    fact = _FACT_LEAD.sub("", m.group(1).strip()).strip()
    fact = " ".join(fact.split())
    if not fact or fact.endswith("?") or _NOT_A_FACT.match(fact):
        return None
    return fact[:MAX_FACT_CHARS]


class MemoryStore:
    def __init__(
        self,
        max_messages: int = MAX_HISTORY_MESSAGES,
        backend: Any = None,
        *,
        compact_batch: int = 10,
    ) -> None:
        self.max_messages = max_messages
        self.backend = backend
        self.compact_batch = compact_batch
        self._msgs: dict[int, list[ChatMessage]] = {}
        self._overflow: dict[int, list[ChatMessage]] = {}
        self._summary: dict[int, str] = {}
        self._facts: dict[int, list[str]] = {}
        self._gen: dict[int, int] = {}
        self._loaded: set[int] = set()
        self._summarizer: Summarizer | None = None
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._compacting: set[int] = set()

    # ------------------------------------------------------------ setup
    def set_summarizer(self, fn: Summarizer | None) -> None:
        self._summarizer = fn

    async def ensure_loaded(self, user_id: int) -> None:
        """Load this user's saved memory once per process (call before use)."""
        if user_id in self._loaded:
            return
        self._loaded.add(user_id)
        if self.backend is None:
            return
        try:
            mem = await self.backend.load(user_id, self.max_messages)
        except Exception as e:
            log.error("memory_load_failed user=%s err=%s", user_id, e)
            return
        saved = [ChatMessage(role=r, content=c) for r, c in mem.messages]  # type: ignore[arg-type]
        self._msgs[user_id] = (saved + self._msgs.get(user_id, []))[-self.max_messages:]
        if mem.summary and not self._summary.get(user_id):
            self._summary[user_id] = mem.summary
        if mem.facts and not self._facts.get(user_id):
            self._facts[user_id] = list(mem.facts)

    # ----------------------------------------------------- sync history API
    def get(self, user_id: int) -> list[ChatMessage]:
        return list(self._msgs.get(user_id, []))

    def size(self, user_id: int) -> int:
        return len(self._msgs.get(user_id, []))

    def add_user(self, user_id: int, text: str) -> None:
        self._add(user_id, "user", text)

    def add_assistant(self, user_id: int, text: str) -> None:
        self._add(user_id, "assistant", text)

    def _add(self, user_id: int, role: str, text: str) -> None:
        msgs = self._msgs.setdefault(user_id, [])
        msgs.append(ChatMessage(role=role, content=text))  # type: ignore[arg-type]
        while len(msgs) > self.max_messages:
            old = self._overflow.setdefault(user_id, [])
            old.append(msgs.pop(0))
            if len(old) > MAX_OVERFLOW:
                del old[: len(old) - MAX_OVERFLOW]
        self._enqueue("append", user_id, role, text)

    def clear(self, user_id: int) -> None:
        """Forget the conversation (messages + summary). Facts are kept."""
        self._msgs.pop(user_id, None)
        self._overflow.pop(user_id, None)
        self._summary.pop(user_id, None)
        self._gen[user_id] = self._gen.get(user_id, 0) + 1  # cancels in-flight compaction
        self._enqueue("clear_history", user_id)
        log.info("memory_cleared user_id=%s", user_id)

    # -------------------------------------------------------------- facts
    def facts(self, user_id: int) -> list[str]:
        return list(self._facts.get(user_id, []))

    def add_fact(self, user_id: int, fact: str) -> bool:
        fact = " ".join((fact or "").split())[:MAX_FACT_CHARS]
        if not fact:
            return False
        facts = self._facts.setdefault(user_id, [])
        if any(f.lower() == fact.lower() for f in facts):
            return False
        if len(facts) >= MAX_FACTS:
            facts.pop(0)
        facts.append(fact)
        self._enqueue("set_facts", user_id, list(facts))
        return True

    def remove_fact(self, user_id: int, index: int) -> str | None:
        """Remove fact number `index` (1-based). Returns it, or None."""
        facts = self._facts.get(user_id, [])
        if not 1 <= index <= len(facts):
            return None
        removed = facts.pop(index - 1)
        self._enqueue("set_facts", user_id, list(facts))
        return removed

    def clear_facts(self, user_id: int) -> int:
        n = len(self._facts.get(user_id, []))
        self._facts[user_id] = []
        self._enqueue("set_facts", user_id, [])
        return n

    # ------------------------------------------------------------ context
    def summary(self, user_id: int) -> str:
        return self._summary.get(user_id, "")

    def context_block(self, user_id: int) -> str:
        """Text for the system prompt: known facts + summary of older chat."""
        parts: list[str] = []
        facts = self._facts.get(user_id) or []
        if facts:
            parts.append(
                "Things the user asked you to remember (treat as true):\n"
                + "\n".join(f"- {f}" for f in facts)
            )
        summ = self._summary.get(user_id)
        if summ:
            parts.append(f"Summary of the earlier part of this conversation:\n{summ}")
        return "\n\n".join(parts)

    # ---------------------------------------------------------- compaction
    def schedule_compaction(self, user_id: int) -> None:
        if (
            self._summarizer is None
            or user_id in self._compacting
            or len(self._overflow.get(user_id, [])) < self.compact_batch
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._compact(user_id))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _compact(self, user_id: int) -> None:
        self._compacting.add(user_id)
        gen = self._gen.get(user_id, 0)
        try:
            batch = list(self._overflow.get(user_id, []))
            if not batch or self._summarizer is None:
                return
            new_summary = await self._summarizer(self._summary.get(user_id, ""), batch)
            if not new_summary or self._gen.get(user_id, 0) != gen:
                return  # failed, or the user reset the chat meanwhile
            self._summary[user_id] = new_summary
            del self._overflow[user_id][: len(batch)]
            self._enqueue("set_summary", user_id, new_summary)
            self._enqueue("trim", user_id, self.max_messages)
        except Exception as e:
            log.warning("memory_compaction_failed user=%s err=%s", user_id, e)
        finally:
            self._compacting.discard(user_id)

    # --------------------------------------------------------- persistence
    def _enqueue(self, *op: Any) -> None:
        if self.backend is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop (plain sync use): RAM only
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._drain())
        self._queue.put_nowait(op)

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            op = await self._queue.get()
            try:
                await self._apply(op)
            except Exception as e:
                log.error("memory_persist_failed op=%s err=%s", op[0], e)
            finally:
                self._queue.task_done()

    async def _apply(self, op: tuple) -> None:
        kind, user_id, *args = op
        b = self.backend
        if kind == "append":
            await b.append(user_id, *args)
        elif kind == "clear_history":
            await b.clear_history(user_id)
        elif kind == "set_facts":
            await b.set_facts(user_id, args[0])
        elif kind == "set_summary":
            await b.set_summary(user_id, args[0])
        elif kind == "trim":
            await b.trim(user_id, args[0])

    async def flush(self) -> None:
        """Wait for compaction and for every queued write to finish."""
        if self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
        if self._queue is not None:
            await self._queue.join()

    async def close(self) -> None:
        await self.flush()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
        if self.backend is not None:
            await self.backend.close()
