"""check_keys.py against a local fake provider (no real network)."""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.services.ai.keypool import KeyPool
from app.services.ai.registry import ModelRegistry
from scripts import check_keys as ck


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        auth = self.headers.get("Authorization", "")
        if auth == "Bearer good":
            self._send(200, {"data": [
                {"id": "models/chat-a"}, {"id": "chat-b"},
                {"id": "text-embed-1"}, {"id": "whisper-x"},
            ]})
        elif auth == "Bearer limited":
            self._send(429, {"error": "slow down"})
        else:
            self._send(401, {"error": "bad key"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send(200, {"choices": []},
                   {"x-ratelimit-limit-requests": "14400",
                    "x-ratelimit-limit-tokens": "6000", "x-other": "no"})


class CheckKeysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.registry = ModelRegistry.from_dict({
            "providers": {"fake": {"base_url": cls.base}},
            "models": [
                {"id": "e-a", "provider": "fake", "model": "chat-a"},
                {"id": "e-gone", "provider": "fake", "model": "retired-model"},
            ],
        })

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _reports(self):
        pool = KeyPool()
        for k in ("good", "limited", "wrongkey"):
            pool.add("fake", k)
        pdef = self.registry.providers["fake"]
        return [ck.check_key(pdef, ks, timeout=5) for ks in pool.keys("fake")]

    def test_statuses_and_model_listing(self):
        good, limited, bad = self._reports()
        self.assertEqual(good.status, "ok")
        self.assertEqual(good.models, ["chat-a", "chat-b", "text-embed-1", "whisper-x"])
        self.assertEqual(limited.status, "limited")
        self.assertEqual(bad.status, "invalid")
        self.assertNotIn("wrongkey", repr(bad))  # never leaks the key

    def test_summary(self):
        s = ck.summarize(self._reports(), self.registry)
        p = s["providers"]["fake"]
        self.assertEqual((p["keys"], p["ok"], p["limited"], p["invalid"]), (3, 1, 1, 1))
        self.assertEqual(p["models_seen"], 4)
        self.assertEqual(p["chat_models_seen"], 2)  # embed + whisper filtered
        self.assertEqual(s["alive_keys"], 2)
        self.assertEqual(s["unique_chat_models_seen"], 2)
        self.assertEqual(s["registry_usable"], ["e-a"])
        self.assertEqual(s["registry_not_visible"], ["e-gone"])

    def test_probe_reads_only_ratelimit_headers(self):
        out = ck.probe_limits(self.registry.providers["fake"], "good", "chat-a", timeout=5)
        self.assertEqual(out["x-ratelimit-limit-requests"], "14400")
        self.assertEqual(out["x-ratelimit-limit-tokens"], "6000")
        self.assertNotIn("x-other", out)

    def test_unreachable_host_is_an_error_not_a_crash(self):
        reg = ModelRegistry.from_dict({
            "providers": {"dead": {"base_url": "http://127.0.0.1:9"}}, "models": []})
        pool = KeyPool()
        pool.add("dead", "k")
        rep = ck.check_key(reg.providers["dead"], pool.keys("dead")[0], timeout=2)
        self.assertEqual(rep.status, "error")
        self.assertTrue(rep.error)

    def test_model_id_helpers(self):
        self.assertEqual(ck.normalize_model_id("models/gemini-x"), "gemini-x")
        self.assertFalse(ck.is_chat_like("text-embedding-004"))
        self.assertTrue(ck.is_chat_like("llama-3.3-70b-versatile"))


if __name__ == "__main__":
    unittest.main()
