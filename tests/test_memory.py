"""Persistent memory: MemoryStore + SQLite backend + Supabase (fake PostgREST)."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.services.ai.models import ChatMessage
from app.services.memory import MemoryStore, extract_fact_request
from app.services.memory_backends import (
    MemoryBackendError,
    SqliteBackend,
    SupabaseBackend,
)


class FactExtractionTests(unittest.TestCase):
    def test_hinglish_and_english(self):
        cases = {
            "yaad rakhna ki mera naam Vikash hai": "mera naam Vikash hai",
            "Yaad rakho: main BCA first year me hu": "main BCA first year me hu",
            "remember that I use Python": "I use Python",
            "please remember I like short answers": "I like short answers",
        }
        for text, expected in cases.items():
            self.assertEqual(extract_fact_request(text), expected, text)

    def test_not_facts(self):
        for text in ["hello", "remember what I said?", "yaad rakhna kya hota hai",
                     "yaad rakhna", "how to remember things"]:
            self.assertIsNone(extract_fact_request(text), text)


class RamStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_api_and_cap(self):
        m = MemoryStore(max_messages=4)
        for i in range(3):
            m.add_user(1, f"q{i}")
            m.add_assistant(1, f"a{i}")
        self.assertEqual(m.size(1), 4)
        self.assertEqual([x.content for x in m.get(1)], ["q1", "a1", "q2", "a2"])
        m.get(1).clear()  # get() returns a copy
        self.assertEqual(m.size(1), 4)
        m.clear(1)
        self.assertEqual(m.size(1), 0)

    async def test_facts_survive_clear_and_show_in_context(self):
        m = MemoryStore()
        self.assertTrue(m.add_fact(1, "naam Vikash hai"))
        self.assertFalse(m.add_fact(1, "NAAM vikash hai"))  # duplicate
        m.add_user(1, "hi")
        m.clear(1)
        self.assertIn("naam Vikash hai", m.context_block(1))
        self.assertEqual(m.remove_fact(1, 1), "naam Vikash hai")
        self.assertIsNone(m.remove_fact(1, 1))
        self.assertEqual(m.context_block(1), "")


class SqlitePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "sub" / "mem.db")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def _store(self, **kw) -> MemoryStore:
        b = SqliteBackend(self.path)
        await b.init()
        return MemoryStore(backend=b, **kw)

    async def test_survives_restart(self):
        m = await self._store()
        await m.ensure_loaded(7)
        m.add_user(7, "mera naam Vikash hai")
        m.add_assistant(7, "Nice to meet you")
        m.add_fact(7, "BCA student")
        await m.close()

        m2 = await self._store()
        await m2.ensure_loaded(7)
        self.assertEqual([x.content for x in m2.get(7)],
                         ["mera naam Vikash hai", "Nice to meet you"])
        self.assertEqual(m2.facts(7), ["BCA student"])
        await m2.ensure_loaded(7)  # idempotent
        self.assertEqual(m2.size(7), 2)
        await m2.close()

    async def test_users_are_isolated(self):
        m = await self._store()
        m.add_user(1, "one")
        m.add_user(2, "two")
        await m.close()
        m2 = await self._store()
        await m2.ensure_loaded(1)
        await m2.ensure_loaded(2)
        self.assertEqual([x.content for x in m2.get(1)], ["one"])
        self.assertEqual([x.content for x in m2.get(2)], ["two"])
        await m2.close()

    async def test_reset_persists_but_facts_stay(self):
        m = await self._store()
        m.add_user(1, "secret chat")
        m.add_fact(1, "likes tea")
        await m.flush()
        m.clear(1)
        await m.close()
        m2 = await self._store()
        await m2.ensure_loaded(1)
        self.assertEqual(m2.size(1), 0)
        self.assertEqual(m2.facts(1), ["likes tea"])
        await m2.close()

    async def test_load_limits_to_last_n(self):
        m = await self._store(max_messages=3)
        for i in range(6):
            m.add_user(1, f"m{i}")
        await m.close()
        m2 = await self._store(max_messages=3)
        await m2.ensure_loaded(1)
        self.assertEqual([x.content for x in m2.get(1)], ["m3", "m4", "m5"])
        await m2.close()

    async def test_compaction_summarises_and_trims(self):
        seen = {}

        async def summarizer(prev, msgs):
            seen["prev"], seen["n"] = prev, len(msgs)
            return "User is Vikash."

        m = await self._store(max_messages=4, compact_batch=3)
        m.set_summarizer(summarizer)
        for i in range(7):
            m.add_user(1, f"m{i}")
        m.schedule_compaction(1)  # 3 overflowed -> triggers
        await m.flush()
        self.assertEqual(seen["n"], 3)
        self.assertEqual(m.summary(1), "User is Vikash.")
        self.assertIn("User is Vikash.", m.context_block(1))
        await m.close()

        m2 = await self._store(max_messages=4)
        await m2.ensure_loaded(1)
        self.assertEqual(m2.summary(1), "User is Vikash.")
        self.assertEqual([x.content for x in m2.get(1)], ["m3", "m4", "m5", "m6"])
        await m2.close()

    async def test_compaction_failure_keeps_overflow(self):
        async def boom(prev, msgs):
            raise RuntimeError("no model")

        m = await self._store(max_messages=2, compact_batch=2)
        m.set_summarizer(boom)
        for i in range(5):
            m.add_user(1, f"m{i}")
        m.schedule_compaction(1)
        await m.flush()
        self.assertEqual(m.summary(1), "")
        self.assertEqual(len(m._overflow[1]), 3)  # retried next time
        await m.close()

    async def test_reset_during_compaction_discards_summary(self):
        import asyncio

        gate = asyncio.Event()

        async def slow(prev, msgs):
            await gate.wait()
            return "stale summary"

        m = await self._store(max_messages=2, compact_batch=2)
        m.set_summarizer(slow)
        for i in range(5):
            m.add_user(1, f"m{i}")
        m.schedule_compaction(1)
        await asyncio.sleep(0)
        m.clear(1)
        gate.set()
        await m.flush()
        self.assertEqual(m.summary(1), "")
        await m.close()

    async def test_backend_failure_does_not_break_chat(self):
        class Broken:
            async def append(self, *a): raise MemoryBackendError("down")
            async def load(self, *a): raise MemoryBackendError("down")
            async def close(self): pass

        m = MemoryStore(backend=Broken())
        await m.ensure_loaded(1)  # swallowed
        m.add_user(1, "still works")
        await m.flush()
        self.assertEqual(m.size(1), 1)


# ---------------------------------------------------------- fake Supabase
class _FakePostgREST(BaseHTTPRequestHandler):
    db: dict = {}
    fail = False

    def log_message(self, *a):  # silence
        pass

    def _reply(self, code, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse(self):
        u = urlparse(self.path)
        table = u.path.rsplit("/", 1)[-1]
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        return table, q

    @staticmethod
    def _match(row, q):
        for col, cond in q.items():
            if col in ("select", "order", "limit", "offset", "on_conflict"):
                continue
            op, _, val = cond.partition(".")
            cur = row.get(col)
            if op == "eq" and str(cur) != val:
                return False
            if op == "lte" and not cur <= int(val):
                return False
        return True

    def do_GET(self):
        if self.fail:
            return self._reply(500, {"message": "boom"})
        if self.headers.get("apikey") != "KEY":
            return self._reply(401, {"message": "bad key"})
        table, q = self._parse()
        rows = [r for r in self.db.setdefault(table, []) if self._match(r, q)]
        if q.get("order") == "id.desc":
            rows.sort(key=lambda r: -r["id"])
        rows = rows[int(q.get("offset", 0)):]
        if "limit" in q:
            rows = rows[: int(q["limit"])]
        cols = q.get("select", "*").split(",")
        if cols != ["*"]:
            rows = [{c: r.get(c) for c in cols} for r in rows]
        self._reply(200, rows)

    def do_POST(self):
        table, q = self._parse()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rows = self.db.setdefault(table, [])
        if table == "assistx_messages":
            for item in body:
                item["id"] = max([r["id"] for r in rows], default=0) + 1
                rows.append(item)
        else:  # profiles upsert (merge-duplicates)
            for r in rows:
                if r["user_id"] == body["user_id"]:
                    r.update(body)
                    break
            else:
                rows.append(body)
        self._reply(201)

    def do_DELETE(self):
        table, q = self._parse()
        rows = self.db.setdefault(table, [])
        self.db[table] = [r for r in rows if not self._match(r, q)]
        self._reply(204)


class SupabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _FakePostgREST.db = {}
        _FakePostgREST.fail = False
        self.server = HTTPServer(("127.0.0.1", 0), _FakePostgREST)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    async def asyncTearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _backend(self, key="KEY"):
        return SupabaseBackend(self.url, key)

    async def test_roundtrip_through_store(self):
        b = self._backend()
        await b.init()
        m = MemoryStore(backend=b)
        m.add_user(5, "hello")
        m.add_assistant(5, "hi!")
        m.add_fact(5, "likes chai")
        await m.flush()

        m2 = MemoryStore(backend=self._backend())
        await m2.ensure_loaded(5)
        self.assertEqual([(x.role, x.content) for x in m2.get(5)],
                         [("user", "hello"), ("assistant", "hi!")])
        self.assertEqual(m2.facts(5), ["likes chai"])
        await m.close()
        await m2.close()

    async def test_clear_trim_and_summary(self):
        b = self._backend()
        for i in range(6):
            await b.append(1, "user", f"m{i}")
        await b.trim(1, 2)
        await b.set_summary(1, "sum")
        mem = await b.load(1, 10)
        self.assertEqual([c for _, c in mem.messages], ["m4", "m5"])
        self.assertEqual(mem.summary, "sum")
        await b.clear_history(1)
        mem = await b.load(1, 10)
        self.assertEqual((mem.messages, mem.summary), ([], ""))

    async def test_errors_are_wrapped(self):
        with self.assertRaises(MemoryBackendError):
            await self._backend(key="WRONG").init()
        _FakePostgREST.fail = True
        with self.assertRaises(MemoryBackendError):
            await self._backend().init()

    def test_requires_credentials(self):
        with self.assertRaises(MemoryBackendError):
            SupabaseBackend("", "")


if __name__ == "__main__":
    unittest.main()
