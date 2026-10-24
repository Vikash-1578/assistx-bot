"""
Model registry, key pool and PoolEngine tests.

No network and no third-party packages: every provider call is a scripted
fake and the quota store is in-memory. Written with unittest so it runs with
`python -m unittest` AND under pytest.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services.ai.exceptions import (
    AllProvidersFailed,
    AuthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from app.services.ai.keypool import KeyPool, parse_keys
from app.services.ai.models import AIResponse, ChatMessage, TaskType
from app.services.ai.pool_engine import PoolEngine, build_pool_engine
from app.services.ai.registry import ModelRegistry, RegistryError

K1 = "gsk_1111aaaa"
K2 = "gsk_2222bbbb"
MSG = [ChatMessage(role="user", content="hi")]

REG = {
    "providers": {
        "groq": {"base_url": "http://x/groq", "daily_quota": 100},
        "gemini": {"base_url": "http://x/gemini", "daily_quota": 100},
    },
    "models": [
        {"id": "g-big", "provider": "groq", "model": "big", "tier": 1, "priority": 10},
        {"id": "g-mid", "provider": "groq", "model": "mid", "tier": 2, "priority": 10},
        {"id": "g-fast", "provider": "groq", "model": "fast", "tier": 3, "priority": 10},
        {"id": "m-mid", "provider": "gemini", "model": "gmid", "tier": 2, "priority": 20},
        {"id": "m-vis", "provider": "gemini", "model": "gvis", "tier": 2,
         "priority": 30, "vision": True},
    ],
}


def fake_settings(**ai_over):
    ai = dict(
        request_timeout=5, failure_threshold=3, cooldown_seconds=60,
        global_concurrency=5, user_concurrency=1, temperature=0.7,
        max_output_tokens=1024, key_rotations=4, keys_file="", keys_inline="",
        models_file="",
    )
    ai.update(ai_over)
    return SimpleNamespace(
        ai=SimpleNamespace(**ai),
        freellmapi=SimpleNamespace(
            enabled=False, base_url="", api_key="", daily_quota=0,
            model_chat="", priority=1,
        ),
        enabled_providers=lambda: {},
    )


class FakeQuota:
    """In-memory stand-in for QuotaTracker (same async interface)."""

    def __init__(self):
        self.req: dict[str, int] = {}
        self.fail: dict[str, int] = {}

    async def snapshot(self, provider, daily_limit):
        return {"requests": self.req.get(provider, 0)}

    async def record_success(self, provider, input_tokens=0, output_tokens=0,
                             total_tokens=0):
        self.req[provider] = self.req.get(provider, 0) + 1

    async def record_failure(self, provider):
        self.fail[provider] = self.fail.get(provider, 0) + 1


class FakeClient:
    def __init__(self, provider, key, script):
        self.provider, self.key, self.script = provider, key, script

    async def generate(self, messages, *, model, task, temperature=0.7,
                       max_tokens=4096):
        return await self.script(self.provider, self.key, model, task)

    async def close(self):
        pass


class Script:
    def __init__(self, fn=None):
        self.fn = fn or (lambda provider, key, model, n: f"{provider}/{model}")
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, provider, key, model, task):
        self.calls.append((provider, key, model))
        r = self.fn(provider, key, model, len(self.calls))
        if isinstance(r, Exception):
            raise r
        return AIResponse(text=r, provider=provider, model=model, total_tokens=5)


def make_engine(keys, script, reg=None, **ai_over):
    pool = KeyPool()
    for p, k in keys:
        pool.add(p, k)
    return PoolEngine(
        fake_settings(**ai_over),
        ModelRegistry.from_dict(reg or REG),
        pool,
        FakeQuota(),
        client_factory=lambda pdef, key: FakeClient(pdef.name, key, script),
    )


class KeyParsingTests(unittest.TestCase):
    def test_parse_keys_comments_aliases_dedupe(self):
        text = """
        # my keys
        groq:gsk_aaa
        google:AIza111   # gemini key
        "hf":hf_xyz, nim:nv_1;or:sk-or-1
        groq:gsk_aaa
        garbage line
        """
        self.assertEqual(parse_keys(text), [
            ("groq", "gsk_aaa"),
            ("gemini", "AIza111"),
            ("huggingface", "hf_xyz"),
            ("nvidia", "nv_1"),
            ("openrouter", "sk-or-1"),
            ("groq", "gsk_aaa"),
        ])
        pool = KeyPool()
        self.assertEqual(pool.add_text(text), 5)  # duplicate dropped
        self.assertEqual(pool.count(), 5)
        self.assertEqual(pool.count("groq"), 1)

    def test_keypool_file_and_cooldown(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "keys.txt"
            f.write_text("groq:k_one\ngroq:k_two\n")
            pool = KeyPool()
            self.assertEqual(pool.add_file(f), 2)
            first = pool.keys("groq")[0]
            pool.mark_rate_limited(first, 30)
            self.assertEqual([k.key for k in pool.available("groq")], ["k_two"])
            pool.disable(pool.keys("groq")[1], "bad")
            self.assertEqual(pool.available("groq"), [])
            pool.mark_ok(first)
            self.assertEqual([k.key for k in pool.available("groq")], ["k_one"])
            self.assertEqual(pool.add_file(Path(d) / "missing.txt"), 0)

    def test_rate_limit_backoff_grows(self):
        pool = KeyPool()
        pool.add("groq", "k_one")
        ks = pool.keys("groq")[0]
        pool.mark_rate_limited(ks, None)
        first = ks.cooldown_until
        pool.mark_rate_limited(ks, None)
        self.assertGreater(ks.cooldown_until, first + 30)  # 60s -> 120s


class RegistryTests(unittest.TestCase):
    def test_default_models_file_is_valid(self):
        reg = ModelRegistry.load()
        self.assertGreaterEqual(len(reg.models), 20)
        self.assertTrue({"groq", "gemini", "nvidia"} <= set(reg.providers))
        self.assertTrue(any(m.vision for m in reg.models))
        self.assertEqual({m.tier for m in reg.models}, {1, 2, 3})

    def test_registry_rejects_bad_files(self):
        p = {"p": {"base_url": "http://x"}}
        bad = [
            {"providers": {}, "models": [
                {"id": "a", "provider": "nope", "model": "m"}]},
            {"providers": p, "models": [
                {"id": "a", "provider": "p", "model": "m"},
                {"id": "a", "provider": "p", "model": "n"}]},
            {"providers": p, "models": [
                {"id": "a", "provider": "p", "model": "m", "tier": 9}]},
            {"providers": p, "models": [
                {"id": "a", "provider": "p", "model": "m", "tasks": ["fly"]}]},
        ]
        for data in bad:
            with self.assertRaises(RegistryError):
                ModelRegistry.from_dict(data)


class PoolEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_tier_preference(self):
        engine = make_engine([("groq", K1)], Script())
        r = await engine.generate(MSG, task=TaskType.CHAT, prefer="balanced")
        self.assertEqual((r.entry_id, r.tier), ("g-mid", 2))
        r = await engine.generate(MSG, task=TaskType.CHAT, prefer="speed")
        self.assertEqual(r.entry_id, "g-fast")
        r = await engine.generate(MSG, task=TaskType.CHAT, prefer="quality")
        self.assertEqual(r.entry_id, "g-big")

    async def test_rate_limit_rotates_to_next_key(self):
        def fn(provider, key, model, n):
            return RateLimitError("groq", "429", retry_after=30) if key == K1 else "ok"

        script = Script(fn)
        engine = make_engine([("groq", K1), ("groq", K2)], script)
        r = await engine.generate(MSG)
        self.assertEqual(r.text, "ok")
        self.assertEqual([c[1] for c in script.calls], [K1, K2])  # same request
        self.assertEqual([k.key for k in engine.keypool.available("groq")], [K2])
        await engine.generate(MSG)
        self.assertEqual([c[1] for c in script.calls][2:], [K2])  # K1 cooling

    async def test_auth_error_disables_key(self):
        script = Script(lambda p, k, m, n: AuthError("groq") if k == K1 else "ok")
        engine = make_engine([("groq", K1), ("groq", K2)], script)
        r = await engine.generate(MSG)
        self.assertEqual(r.text, "ok")
        bad = engine.keypool.keys("groq")[0]
        self.assertTrue(bad.disabled_reason)
        self.assertFalse(bad.available())

    async def test_dead_model_is_disabled_and_skipped(self):
        script = Script(
            lambda p, k, m, n: ModelNotFoundError(p, m) if m == "mid" else f"{p}/{m}"
        )
        engine = make_engine([("groq", K1), ("gemini", "AIza_1")], script)
        r = await engine.generate(MSG)
        self.assertEqual(r.entry_id, "m-mid")  # groq mid died -> next tier-2 model
        self.assertIn("g-mid", engine._entry_disabled)
        n = len(script.calls)
        await engine.generate(MSG)
        self.assertTrue(all(c[2] != "mid" for c in script.calls[n:]))

    async def test_fallback_across_providers(self):
        script = Script(
            lambda p, k, m, n: ProviderError("groq", "boom") if p == "groq" else "from-gemini"
        )
        engine = make_engine([("groq", K1), ("gemini", "AIza_1")], script)
        r = await engine.generate(MSG)
        self.assertEqual((r.text, r.provider), ("from-gemini", "gemini"))

    async def test_empty_reply_counts_as_failure(self):
        script = Script(lambda p, k, m, n: "   " if p == "groq" else "real answer")
        engine = make_engine([("groq", K1), ("gemini", "AIza_1")], script)
        r = await engine.generate(MSG)
        self.assertEqual(r.text, "real answer")

    async def test_entry_cooldown_after_repeated_failures(self):
        script = Script(
            lambda p, k, m, n: ProviderError(p, "boom") if m == "mid" else f"{p}/{m}"
        )
        engine = make_engine([("groq", K1)], script, failure_threshold=2)
        for _ in range(2):
            await engine.generate(MSG)
        n = len(script.calls)
        await engine.generate(MSG)
        self.assertTrue(all(c[2] != "mid" for c in script.calls[n:]))

    async def test_all_fail_raises(self):
        script = Script(lambda *a: ProviderError("x", "boom"))
        engine = make_engine([("groq", K1)], script)
        with self.assertRaises(AllProvidersFailed):
            await engine.generate(MSG)

    async def test_exclude_skips_entries(self):
        engine = make_engine([("groq", K1)], Script())
        r = await engine.generate(MSG, prefer="balanced", exclude={"g-mid"})
        self.assertEqual(r.entry_id, "g-big")

    async def test_vision_only_uses_vision_models(self):
        script = Script()
        engine = make_engine([("groq", K1), ("gemini", "AIza_1")], script)
        r = await engine.generate(MSG, task=TaskType.VISION)
        self.assertEqual(r.entry_id, "m-vis")
        self.assertEqual({c[2] for c in script.calls}, {"gvis"})

    async def test_daily_cap_per_key(self):
        reg = {
            "providers": {"groq": {"base_url": "http://x", "daily_quota": 2}},
            "models": [{"id": "only", "provider": "groq", "model": "m"}],
        }
        engine = make_engine([("groq", K1)], Script(), reg)
        await engine.generate(MSG)
        await engine.generate(MSG)
        with self.assertRaises(AllProvidersFailed):
            await engine.generate(MSG)

    async def test_load_spreads_across_keys(self):
        script = Script()
        engine = make_engine([("groq", K1), ("groq", K2)], script)
        for _ in range(4):
            await engine.generate(MSG, prefer="speed")
        used = [c[1] for c in script.calls]
        self.assertEqual((used.count(K1), used.count(K2)), (2, 2))

    async def test_max_tokens_capped_per_entry(self):
        seen = []

        class Capture(FakeClient):
            async def generate(self, messages, *, model, task, temperature=0.7,
                               max_tokens=4096):
                seen.append(max_tokens)
                return AIResponse(text="ok", provider="groq", model=model)

        reg = {
            "providers": {"groq": {"base_url": "http://x"}},
            "models": [{"id": "e", "provider": "groq", "model": "m", "max_tokens": 300}],
        }
        pool = KeyPool()
        pool.add("groq", K1)
        engine = PoolEngine(
            fake_settings(), ModelRegistry.from_dict(reg), pool, FakeQuota(),
            client_factory=lambda pdef, key: Capture(pdef.name, key, None),
        )
        await engine.generate(MSG, max_tokens=4000)
        self.assertEqual(seen, [300])

    async def test_status_and_summary(self):
        engine = make_engine(
            [("groq", K1), ("groq", K2), ("gemini", "AIza_1")], Script()
        )
        await engine.generate(MSG)
        st = {s.name: s for s in await engine.status()}
        self.assertEqual(set(st), {"groq", "gemini"})
        self.assertTrue(st["groq"].available)
        self.assertTrue(st["groq"].detail.startswith("2/2 keys ok"))
        self.assertEqual(st["groq"].daily_limit, 200)
        self.assertEqual(st["groq"].requests_today, 1)
        summ = engine.summary()
        self.assertEqual((summ["keys"], summ["models"]), (3, 5))
        self.assertEqual(summ["by_provider"]["groq"], {"keys": 2, "models": 3})


class BuildTests(unittest.TestCase):
    def test_build_from_inline_keys(self):
        engine = build_pool_engine(
            fake_settings(keys_inline=f"groq:{K1}"), FakeQuota()
        )
        s = engine.summary()
        self.assertEqual(s["keys"], 1)
        self.assertGreaterEqual(s["models"], 4)  # all groq entries in models.json

    def test_build_from_keys_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "keys.txt"
            f.write_text(f"groq:{K1}\ngemini:AIza_x\n")
            engine = build_pool_engine(
                fake_settings(keys_file=str(f)), FakeQuota()
            )
        self.assertEqual(engine.summary()["keys"], 2)

    def test_unknown_provider_only_is_rejected(self):
        with self.assertRaises(RegistryError):
            build_pool_engine(fake_settings(keys_inline="foo:abc123"), FakeQuota())

    def test_missing_models_file_is_rejected(self):
        with self.assertRaises(RegistryError):
            build_pool_engine(
                fake_settings(keys_inline=f"groq:{K1}", models_file="/nope.json"),
                FakeQuota(),
            )


if __name__ == "__main__":
    unittest.main()
