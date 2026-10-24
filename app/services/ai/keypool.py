"""
KeyPool — holds MANY API keys (50+) and hands out the healthy ones.

Key sources (all merged, duplicates dropped):
  1. keys file   (default `keys.txt`, one `provider:key` per line, # comments)
  2. AI_KEYS env (same `provider:key` items separated by newline , or ;)
  3. legacy single-key env vars (GROQ_API_KEY, ...) via Settings

Keys are never logged in full; we only use a short fingerprint.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings

log = get_logger(__name__)

PROVIDER_ALIASES = {
    "google": "gemini",
    "gemini": "gemini",
    "hf": "huggingface",
    "huggingface": "huggingface",
    "nim": "nvidia",
    "nvidia": "nvidia",
    "or": "openrouter",
    "openrouter": "openrouter",
}

_COMMENT = re.compile(r"(^|\s)#.*$")


def normalize_provider(name: str) -> str:
    n = name.strip().lower()
    return PROVIDER_ALIASES.get(n, n)


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 8 else "…"


@dataclass
class KeyState:
    provider: str
    key: str
    fp: str
    cooldown_until: float = 0.0
    strikes: int = 0
    disabled_reason: str = ""

    @property
    def usage_name(self) -> str:
        return f"{self.provider}:{self.fp}"

    def available(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return not self.disabled_reason and self.cooldown_until <= now


def parse_keys(text: str) -> list[tuple[str, str]]:
    """Parse `provider:key` items. Bad lines are skipped, never raise."""
    out: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = _COMMENT.sub("", line).strip()
        if not line:
            continue
        for piece in re.split(r"[;,]", line):
            piece = piece.strip().strip("\"'")
            if ":" not in piece:
                continue
            provider, _, key = piece.partition(":")
            provider = normalize_provider(provider.strip().strip("\"'"))
            key = key.strip().strip("\"'")
            if provider and key:
                out.append((provider, key))
    return out


class KeyPool:
    def __init__(self) -> None:
        self._keys: dict[str, list[KeyState]] = {}

    # ---- loading --------------------------------------------------------
    def add(self, provider: str, key: str) -> bool:
        provider = normalize_provider(provider)
        key = key.strip()
        if not provider or not key:
            return False
        fp = fingerprint(key)
        bucket = self._keys.setdefault(provider, [])
        if any(k.fp == fp for k in bucket):
            return False
        bucket.append(KeyState(provider=provider, key=key, fp=fp))
        return True

    def add_text(self, text: str) -> int:
        return sum(1 for p, k in parse_keys(text) if self.add(p, k))

    def add_file(self, path: str | Path) -> int:
        p = Path(path)
        if not p.is_file():
            return 0
        return self.add_text(p.read_text(encoding="utf-8", errors="ignore"))

    @classmethod
    def from_settings(cls, settings: "Settings") -> "KeyPool":
        pool = cls()
        n_file = pool.add_file(settings.ai.keys_file) if settings.ai.keys_file else 0
        n_inline = pool.add_text(settings.ai.keys_inline)
        n_legacy = 0
        for name, cfg in settings.enabled_providers().items():
            if name == "freellmapi":  # local gateway, handled separately
                continue
            if cfg.enabled and cfg.api_key:
                if pool.add(name, cfg.api_key):
                    n_legacy += 1
        log.info(
            "keypool_loaded file=%s inline=%s legacy=%s total=%s",
            n_file, n_inline, n_legacy, pool.count(),
        )
        return pool

    # ---- querying -------------------------------------------------------
    def providers(self) -> set[str]:
        return {p for p, ks in self._keys.items() if ks}

    def keys(self, provider: str) -> list[KeyState]:
        return list(self._keys.get(provider, []))

    def count(self, provider: str | None = None) -> int:
        if provider:
            return len(self._keys.get(provider, []))
        return sum(len(v) for v in self._keys.values())

    def available(self, provider: str, now: float | None = None) -> list[KeyState]:
        now = time.monotonic() if now is None else now
        return [k for k in self._keys.get(provider, []) if k.available(now)]

    # ---- health feedback ------------------------------------------------
    def mark_rate_limited(self, ks: KeyState, retry_after: float | None) -> None:
        ks.strikes += 1
        if retry_after is not None:
            wait = max(1.0, min(float(retry_after), 3600.0))
        else:
            wait = min(3600.0, 60.0 * (2 ** (ks.strikes - 1)))
        ks.cooldown_until = time.monotonic() + wait
        log.warning(
            "key_rate_limited provider=%s key=%s cooldown=%.0fs",
            ks.provider, mask(ks.key), wait,
        )

    def mark_ok(self, ks: KeyState) -> None:
        ks.strikes = 0
        ks.cooldown_until = 0.0

    def disable(self, ks: KeyState, reason: str) -> None:
        ks.disabled_reason = reason
        log.error(
            "key_disabled provider=%s key=%s reason=%s",
            ks.provider, mask(ks.key), reason,
        )
