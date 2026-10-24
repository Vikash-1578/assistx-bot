#!/usr/bin/env python3
"""
Check every API key you own.

  python scripts/check_keys.py                  # keys from keys.txt / AI_KEYS
  python scripts/check_keys.py --keys my.txt --probe --json report.json

For each key it tells you: alive / invalid / rate-limited, which models the key
can see, and which entries of models.json your keys can really use.
`--probe` sends ONE tiny chat request per provider to read the rate-limit
headers the provider returns (costs a handful of tokens).

Only the standard library is used. Keys are never printed in full.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ai.keypool import KeyPool, KeyState, mask  # noqa: E402
from app.services.ai.registry import (  # noqa: E402
    DEFAULT_MODELS_FILE,
    ModelRegistry,
    ProviderDef,
)

NON_CHAT_HINTS = (
    "embed", "whisper", "tts", "guard", "moderation", "rerank", "dall-e",
    "imagen", "stable-diffusion", "speech", "transcribe", "audio", "veo",
    "-image", "ocr",
)


@dataclass
class KeyReport:
    provider: str
    key_hint: str
    status: str  # ok | limited | invalid | error
    http: int = 0
    models: list[str] = field(default_factory=list)
    error: str = ""


def _request(url, key, headers, timeout, body=None):
    hdrs = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "assistx-check-keys/1.0",
        **(headers or {}),
    }
    req = urllib.request.Request(
        url, data=body, headers=hdrs, method="POST" if body is not None else "GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        hdr = {k.lower(): v for k, v in (e.headers or {}).items()}
        return e.code, hdr, e.read()


def normalize_model_id(model_id: str) -> str:
    return model_id[len("models/"):] if model_id.startswith("models/") else model_id


def is_chat_like(model_id: str) -> bool:
    m = model_id.lower()
    return not any(h in m for h in NON_CHAT_HINTS)


def check_key(pdef: ProviderDef, ks: KeyState, timeout: float = 20.0) -> KeyReport:
    rep = KeyReport(provider=pdef.name, key_hint=mask(ks.key), status="error")
    try:
        code, _hdr, body = _request(
            f"{pdef.base_url}/models", ks.key, pdef.headers, timeout
        )
    except Exception as e:  # network / DNS / timeout
        rep.error = f"{type(e).__name__}: {e}"[:140]
        return rep
    rep.http = code
    if code == 200:
        try:
            data = json.loads(body.decode("utf-8", "ignore"))
            items = data.get("data") if isinstance(data, dict) else data
            rep.models = sorted(
                {
                    normalize_model_id(str(i["id"]))
                    for i in (items or [])
                    if isinstance(i, dict) and i.get("id")
                }
            )
            rep.status = "ok"
        except Exception as e:
            rep.error = f"bad JSON: {e}"[:140]
    elif code in (401, 403):
        rep.status = "invalid"
    elif code == 429:
        rep.status = "limited"  # key works, just throttled right now
    else:
        rep.error = f"http {code}"
    return rep


def probe_limits(pdef: ProviderDef, key: str, model: str, timeout: float = 30.0) -> dict:
    """One 1-token chat request; returns the rate-limit headers seen."""
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    ).encode()
    try:
        code, hdr, _ = _request(
            f"{pdef.base_url}/chat/completions", key, pdef.headers, timeout, body
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:140]}
    out = {k: v for k, v in hdr.items() if k.startswith("x-ratelimit") or k == "retry-after"}
    out["http"] = code
    return out


def summarize(reports: list[KeyReport], registry: ModelRegistry) -> dict:
    by_provider: dict[str, dict] = {}
    for r in reports:
        p = by_provider.setdefault(
            r.provider,
            {"keys": 0, "ok": 0, "limited": 0, "invalid": 0, "error": 0, "models": set()},
        )
        p["keys"] += 1
        p[r.status] += 1
        p["models"].update(r.models)

    missing: list[str] = []
    usable: list[str] = []
    for e in registry.models:
        p = by_provider.get(e.provider)
        if not p or (p["ok"] + p["limited"]) == 0:
            continue
        # a throttled-only provider can't list models; treat its entries as usable
        if p["ok"] == 0 or e.model in p["models"]:
            usable.append(e.id)
        else:
            missing.append(e.id)

    total_models = set()
    for p in by_provider.values():
        total_models |= {m for m in p["models"] if is_chat_like(m)}
    return {
        "providers": {
            name: {
                **{k: v for k, v in p.items() if k != "models"},
                "models_seen": len(p["models"]),
                "chat_models_seen": sum(1 for m in p["models"] if is_chat_like(m)),
            }
            for name, p in sorted(by_provider.items())
        },
        "total_keys": len(reports),
        "alive_keys": sum(1 for r in reports if r.status in ("ok", "limited")),
        "unique_chat_models_seen": len(total_models),
        "registry_usable": usable,
        "registry_not_visible": missing,
    }


def build_pool(keys_file: str | None) -> KeyPool:
    pool = KeyPool()
    if keys_file:
        pool.add_file(keys_file)
    pool.add_text(os.environ.get("AI_KEYS", ""))
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keys", default="keys.txt", help="keys file (provider:key per line)")
    ap.add_argument("--models", default="", help="models.json (default: bundled)")
    ap.add_argument("--probe", action="store_true", help="read rate-limit headers")
    ap.add_argument("--json", default="", help="write full report to this file")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    try:  # optional: pick up AI_KEYS from .env
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    registry = ModelRegistry.load(args.models or DEFAULT_MODELS_FILE)
    pool = build_pool(args.keys)
    if pool.count() == 0:
        print(f"No keys found. Put `provider:key` lines in {args.keys} or set AI_KEYS.")
        return 1

    jobs = []
    for prov in sorted(pool.providers()):
        pdef = registry.providers.get(prov)
        if pdef is None:
            print(f"! skipping {pool.count(prov)} key(s) of unknown provider '{prov}' "
                  f"(add it to models.json)")
            continue
        jobs.extend((pdef, ks) for ks in pool.keys(prov))

    print(f"Checking {len(jobs)} keys...\n")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        reports = list(ex.map(lambda j: check_key(*j), jobs))

    summary = summarize(reports, registry)
    icon = {"ok": "OK ", "limited": "LIM", "invalid": "BAD", "error": "ERR"}
    for r in reports:
        extra = f"{len(r.models)} models" if r.status == "ok" else (r.error or f"http {r.http}")
        print(f"  [{icon[r.status]}] {r.provider:<12} {r.key_hint:<8} {extra}")

    print("\nPer provider:")
    for name, p in summary["providers"].items():
        print(f"  {name:<12} keys={p['keys']} ok={p['ok']} limited={p['limited']} "
              f"invalid={p['invalid']} error={p['error']}  "
              f"models seen={p['models_seen']} (chat-like {p['chat_models_seen']})")

    print(f"\nAlive keys: {summary['alive_keys']} / {summary['total_keys']}")
    print(f"Unique chat-like models your keys can see: {summary['unique_chat_models_seen']}")
    print(f"models.json entries usable: {len(summary['registry_usable'])}")
    if summary["registry_not_visible"]:
        print("models.json entries NOT visible to your keys (renamed, gated or retired):")
        for mid in summary["registry_not_visible"]:
            print(f"  - {mid}")

    probes: dict[str, dict] = {}
    if args.probe:
        print("\nRate-limit headers (one tiny request per provider):")
        for name in summary["providers"]:
            ok_key = next((r for r in reports if r.provider == name and r.status == "ok"), None)
            entry = next((e for e in registry.models
                          if e.provider == name and e.id in summary["registry_usable"]), None)
            if not ok_key or not entry:
                continue
            ks = next(k for k in pool.keys(name) if mask(k.key) == ok_key.key_hint)
            probes[name] = probe_limits(registry.providers[name], ks.key, entry.model)
            print(f"  {name:<12} {probes[name]}")
        print("  (units depend on the provider: Groq reports requests/DAY and tokens/MINUTE;\n"
              "   monthly token totals are only visible in each provider's dashboard.)")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"summary": summary, "keys": [asdict(r) for r in reports], "probes": probes},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
