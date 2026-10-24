"""
Model registry — the single place that says which models exist.

Adding a model = adding one entry to `app/config_data/models.json`.
No code change needed.

Tiers (how the router picks a model for a question):
    1 = strongest (big / reasoning models)   -> hard questions
    2 = solid all-rounders (70B class)       -> normal questions
    3 = fast / small                         -> greetings, quick replies
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALL_TEXT_TASKS = frozenset(
    {"chat", "code", "content", "freelance", "reasoning", "summary"}
)
VALID_TASKS = ALL_TEXT_TASKS | {"vision"}

DEFAULT_MODELS_FILE = (
    Path(__file__).resolve().parents[2] / "config_data" / "models.json"
)


class RegistryError(ValueError):
    """models.json is invalid or nothing usable is configured."""


@dataclass(frozen=True)
class ProviderDef:
    name: str
    base_url: str
    daily_quota: int = 1000  # requests per KEY per day (our own safety cap)
    headers: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    provider: str
    model: str
    tier: int = 2
    tasks: frozenset[str] = ALL_TEXT_TASKS
    vision: bool = False
    priority: int = 100  # lower = tried first inside the same tier
    enabled: bool = True
    max_tokens: int | None = None


class ModelRegistry:
    def __init__(
        self,
        providers: dict[str, ProviderDef] | None = None,
        models: list[ModelEntry] | None = None,
    ) -> None:
        self.providers: dict[str, ProviderDef] = dict(providers or {})
        self.models: list[ModelEntry] = []
        for m in models or []:
            self.add_model(m)

    # ---- building -------------------------------------------------------
    def add_provider(self, p: ProviderDef) -> None:
        self.providers[p.name] = p

    def add_model(self, m: ModelEntry) -> None:
        if any(x.id == m.id for x in self.models):
            raise RegistryError(f"duplicate model id: {m.id}")
        if m.tier not in (1, 2, 3):
            raise RegistryError(f"{m.id}: tier must be 1, 2 or 3")
        bad = set(m.tasks) - VALID_TASKS
        if bad:
            raise RegistryError(f"{m.id}: unknown tasks {sorted(bad)}")
        self.models.append(m)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRegistry":
        reg = cls()
        for name, p in (data.get("providers") or {}).items():
            if not p.get("base_url"):
                raise RegistryError(f"provider {name}: base_url missing")
            reg.add_provider(
                ProviderDef(
                    name=name.lower(),
                    base_url=str(p["base_url"]).rstrip("/"),
                    daily_quota=int(p.get("daily_quota", 1000)),
                    headers=dict(p.get("headers") or {}),
                )
            )
        for raw in data.get("models") or []:
            try:
                provider = str(raw["provider"]).lower()
                entry = ModelEntry(
                    id=str(raw["id"]),
                    provider=provider,
                    model=str(raw["model"]),
                    tier=int(raw.get("tier", 2)),
                    tasks=frozenset(raw.get("tasks") or ALL_TEXT_TASKS),
                    vision=bool(raw.get("vision", False)),
                    priority=int(raw.get("priority", 100)),
                    enabled=bool(raw.get("enabled", True)),
                    max_tokens=raw.get("max_tokens"),
                )
            except KeyError as e:
                raise RegistryError(f"model entry missing field {e}: {raw}")
            if provider not in reg.providers:
                raise RegistryError(
                    f"{entry.id}: provider '{provider}' not defined in providers"
                )
            reg.add_model(entry)
        return reg

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelRegistry":
        p = Path(path) if path else DEFAULT_MODELS_FILE
        if not p.exists():
            raise RegistryError(f"models file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RegistryError(f"{p}: invalid JSON: {e}")
        return cls.from_dict(data)

    # ---- querying -------------------------------------------------------
    def get(self, entry_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == entry_id), None)

    def candidates(
        self,
        task: str,
        *,
        vision: bool = False,
        providers: set[str] | None = None,
    ) -> list[ModelEntry]:
        pool = [
            m
            for m in self.models
            if m.enabled and (providers is None or m.provider in providers)
        ]
        if vision:
            return [m for m in pool if m.vision]
        matched = [m for m in pool if task in m.tasks]
        if matched:
            return matched
        return [m for m in pool if "chat" in m.tasks]

    def provider_names(self) -> set[str]:
        return set(self.providers)
