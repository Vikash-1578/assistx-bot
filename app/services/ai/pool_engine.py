"""
PoolEngine — routes every request across MANY models and MANY API keys.

Same public surface as AIEngine (`generate`, `status`, `close`) so the rest
of the app does not care which engine it talks to.

How a request is served:
  1. Candidate models = registry entries for the task (vision -> vision models).
  2. Ordered by tier preference:
        prefer="quality"  -> tier 1, 2, 3   (hard questions, escalations)
        prefer="balanced" -> tier 2, 1, 3   (normal questions)
        prefer="speed"    -> tier 3, 2, 1   (greetings, quick replies)
     then by model priority, then by how lightly used its keys are today.
  3. For each model: pick the least-used healthy key. Rate limit -> that key
     cools down and we immediately rotate to the next key. Bad key -> disabled.
     Dead model -> disabled. Other errors -> next model.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable

from app.services.ai.exceptions import (
    AllProvidersFailed,
    AuthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
)
from app.services.ai.keypool import KeyPool, KeyState, mask
from app.services.ai.models import AIResponse, ChatMessage, ProviderStatus, TaskType
from app.services.ai.registry import (
    DEFAULT_MODELS_FILE,
    ModelEntry,
    ModelRegistry,
    ProviderDef,
    RegistryError,
)
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings
    from app.services.ai.quota import QuotaTracker

log = get_logger(__name__)

TIER_RANK: dict[str, dict[int, int]] = {
    "quality": {1: 0, 2: 1, 3: 2},
    "balanced": {2: 0, 1: 1, 3: 2},
    "speed": {3: 0, 2: 1, 1: 2},
}

ClientFactory = Callable[[ProviderDef, str], Any]


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class PoolEngine:
    supports_tiers = True

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        keypool: KeyPool,
        quota: QuotaTracker,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.settings = settings
        self.ai_cfg = settings.ai
        self.registry = registry
        self.keypool = keypool
        self.quota = quota
        self._client_factory = client_factory or self._default_client
        self._clients: dict[str, Any] = {}

        self._entry_fails: dict[str, int] = {}
        self._entry_cooldown: dict[str, float] = {}
        self._entry_disabled: dict[str, str] = {}
        self._entry_last_error: dict[str, str] = {}
        self._usage: dict[str, tuple[str, int]] = {}

        self._global_sem = asyncio.Semaphore(self.ai_cfg.global_concurrency)
        self._user_sems: dict[int, asyncio.Semaphore] = {}

    # ------------------------------------------------------------------ infra
    def _default_client(self, pdef: ProviderDef, key: str) -> Any:
        from app.services.ai.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )

        return OpenAICompatibleProvider(
            name=pdef.name,
            api_key=key,
            base_url=pdef.base_url,
            timeout=self.ai_cfg.request_timeout,
            extra_headers=pdef.headers or None,
        )

    def _client_for(self, provider: str, ks: KeyState) -> Any:
        client = self._clients.get(ks.usage_name)
        if client is None:
            client = self._client_factory(self.registry.providers[provider], ks.key)
            self._clients[ks.usage_name] = client
        return client

    async def close(self) -> None:
        for name, client in self._clients.items():
            try:
                await client.close()
            except Exception as e:  # pragma: no cover
                log.warning("client_close_error client=%s err=%s", name, e)
        self._clients.clear()

    def _user_sem(self, user_id: int) -> asyncio.Semaphore:
        sem = self._user_sems.get(user_id)
        if sem is None:
            sem = asyncio.Semaphore(self.ai_cfg.user_concurrency)
            self._user_sems[user_id] = sem
        return sem

    # ------------------------------------------------------------ key / usage
    async def _requests_today(self, ks: KeyState) -> int:
        cached = self._usage.get(ks.usage_name)
        today = _today()
        if cached is None or cached[0] != today:
            snap = await self.quota.snapshot(ks.usage_name, 1)
            cached = (today, int(snap["requests"]))
            self._usage[ks.usage_name] = cached
        return cached[1]

    def _bump_usage(self, ks: KeyState) -> None:
        today = _today()
        cached = self._usage.get(ks.usage_name)
        base = cached[1] if cached and cached[0] == today else 0
        self._usage[ks.usage_name] = (today, base + 1)

    async def _best_key(
        self, provider: str, tried: set[str] | None = None
    ) -> tuple[KeyState | None, float]:
        """Least-used healthy key under its daily cap, plus its load (0..1)."""
        pdef = self.registry.providers[provider]
        best: KeyState | None = None
        best_frac = 2.0
        for ks in self.keypool.available(provider):
            if tried and ks.fp in tried:
                continue
            used = await self._requests_today(ks)
            if used >= pdef.daily_quota:
                continue
            frac = used / max(1, pdef.daily_quota)
            if frac < best_frac:
                best, best_frac = ks, frac
        return best, best_frac

    # ------------------------------------------------------------- selection
    async def _ordered_entries(
        self,
        task: TaskType,
        vision: bool,
        prefer: str,
        exclude: set[str],
    ) -> list[ModelEntry]:
        now = time.monotonic()
        rank = TIER_RANK.get(prefer, TIER_RANK["balanced"])
        scored: list[tuple[tuple, ModelEntry]] = []
        for e in self.registry.candidates(
            task.value, vision=vision, providers=self.keypool.providers()
        ):
            if e.id in exclude or e.id in self._entry_disabled:
                continue
            if self._entry_cooldown.get(e.id, 0.0) > now:
                continue
            ks, load = await self._best_key(e.provider)
            if ks is None:
                continue
            scored.append(((rank.get(e.tier, 9), e.priority, load, e.id), e))
        scored.sort(key=lambda x: x[0])
        return [e for _, e in scored]

    # --------------------------------------------------------------- failures
    async def _entry_failed(
        self, entry: ModelEntry, err: Exception, cooldown: float | None = None
    ) -> None:
        self._entry_last_error[entry.id] = str(err)[:160]
        await self.quota.record_failure(entry.provider)
        if cooldown is not None:
            self._entry_cooldown[entry.id] = time.monotonic() + cooldown
            self._entry_fails[entry.id] = 0
            log.warning("entry_cooldown entry=%s secs=%s err=%s", entry.id, cooldown, err)
            return
        n = self._entry_fails.get(entry.id, 0) + 1
        self._entry_fails[entry.id] = n
        if n >= self.ai_cfg.failure_threshold:
            self._entry_cooldown[entry.id] = (
                time.monotonic() + self.ai_cfg.cooldown_seconds
            )
            self._entry_fails[entry.id] = 0
            log.warning(
                "entry_cooldown entry=%s secs=%s err=%s",
                entry.id, self.ai_cfg.cooldown_seconds, err,
            )

    async def _try_entry(
        self,
        entry: ModelEntry,
        messages: list[ChatMessage],
        *,
        task: TaskType,
        user_id: int | None,
        temperature: float,
        max_tokens: int,
        attempted: list[str],
    ) -> AIResponse | None:
        tried: set[str] = set()
        if entry.max_tokens:
            max_tokens = min(max_tokens, entry.max_tokens)
        for _ in range(max(1, self.ai_cfg.key_rotations)):
            ks, _load = await self._best_key(entry.provider, tried)
            if ks is None:
                return None
            tried.add(ks.fp)
            attempted.append(f"{entry.id}{mask(ks.key)}")
            client = self._client_for(entry.provider, ks)
            log.info(
                "ai_request entry=%s provider=%s model=%s task=%s user=%s key=%s",
                entry.id, entry.provider, entry.model, task.value, user_id, mask(ks.key),
            )
            try:
                resp: AIResponse = await asyncio.wait_for(
                    client.generate(
                        messages,
                        model=entry.model,
                        task=task,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=self.ai_cfg.request_timeout + 15,
                )
            except RateLimitError as e:
                self.keypool.mark_rate_limited(ks, e.retry_after)
                continue  # rotate to the next key
            except QuotaExhausted:
                self.keypool.mark_rate_limited(ks, 3600.0)
                continue
            except AuthError as e:
                self.keypool.disable(ks, f"auth failed ({e})")
                continue
            except ModelNotFoundError as e:
                self._entry_disabled[entry.id] = "model not found"
                log.error("entry_disabled entry=%s reason=%s", entry.id, e)
                return None
            except InvalidRequestError as e:
                await self._entry_failed(entry, e, cooldown=600.0)
                return None
            except (ProviderError, asyncio.TimeoutError) as e:
                await self._entry_failed(entry, e)
                return None
            except Exception as e:  # unexpected (network, parsing, ...)
                await self._entry_failed(entry, e)
                return None

            if not (resp.text or "").strip():
                await self._entry_failed(entry, ProviderError(entry.provider, "empty reply"))
                return None

            self.keypool.mark_ok(ks)
            self._entry_fails[entry.id] = 0
            self._bump_usage(ks)
            await self.quota.record_success(
                ks.usage_name,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens,
            )
            resp.entry_id = entry.id
            resp.tier = entry.tier
            resp.provider = entry.provider
            log.info(
                "ai_response entry=%s model=%s latency_ms=%s tokens=%s",
                entry.id, resp.model, resp.latency_ms, resp.total_tokens,
            )
            return resp
        return None

    # -------------------------------------------------------------- public API
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        task: TaskType = TaskType.CHAT,
        user_id: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prefer: str = "balanced",
        exclude: Iterable[str] | None = None,
    ) -> AIResponse:
        temperature = self.ai_cfg.temperature if temperature is None else temperature
        max_tokens = self.ai_cfg.max_output_tokens if max_tokens is None else max_tokens

        entries = await self._ordered_entries(
            task, task == TaskType.VISION, prefer, set(exclude or ())
        )
        if not entries:
            raise AllProvidersFailed(["<none available>"])

        attempted: list[str] = []
        async with self._global_sem:
            user_sem = self._user_sem(user_id) if user_id is not None else None
            if user_sem:
                await user_sem.acquire()
            try:
                for entry in entries:
                    resp = await self._try_entry(
                        entry,
                        messages,
                        task=task,
                        user_id=user_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        attempted=attempted,
                    )
                    if resp is not None:
                        return resp
            finally:
                if user_sem:
                    user_sem.release()
        raise AllProvidersFailed(attempted)

    def _usable_entries(self, provider: str) -> list[ModelEntry]:
        return [
            e
            for e in self.registry.models
            if e.provider == provider and e.enabled and e.id not in self._entry_disabled
        ]

    async def status(self) -> list[ProviderStatus]:
        now = time.monotonic()
        out: list[ProviderStatus] = []
        for prov in sorted(self.keypool.providers()):
            pdef = self.registry.providers.get(prov)
            if pdef is None:
                continue
            keys = self.keypool.keys(prov)
            used = 0
            avail = 0
            for ks in keys:
                n = await self._requests_today(ks)
                used += n
                if ks.available(now) and n < pdef.daily_quota:
                    avail += 1
            live = [k for k in keys if not k.disabled_reason]
            cd_left = 0
            if avail == 0 and live:
                cd_left = max(0, int(min(k.cooldown_until for k in live) - now))
            entries = self._usable_entries(prov)
            fails = sum(self._entry_fails.get(e.id, 0) for e in entries)
            out.append(
                ProviderStatus(
                    name=prov,
                    enabled=True,
                    available=avail > 0 and bool(entries),
                    requests_today=used,
                    daily_limit=pdef.daily_quota * len(keys),
                    consecutive_failures=fails,
                    cooldown_seconds_left=cd_left,
                    detail=f"{avail}/{len(keys)} keys ok, {len(entries)} models",
                )
            )
        return out

    def summary(self) -> dict[str, Any]:
        by_provider: dict[str, dict[str, int]] = {}
        total_models = 0
        for prov in sorted(self.keypool.providers()):
            if prov not in self.registry.providers:
                continue
            n_models = len(self._usable_entries(prov))
            total_models += n_models
            by_provider[prov] = {"keys": self.keypool.count(prov), "models": n_models}
        return {
            "keys": self.keypool.count(),
            "models": total_models,
            "by_provider": by_provider,
        }


def build_pool_engine(
    settings: Settings,
    quota: QuotaTracker,
    client_factory: ClientFactory | None = None,
) -> PoolEngine:
    """Load registry + keys from settings. Raises RegistryError if unusable."""
    registry = ModelRegistry.load(settings.ai.models_file or DEFAULT_MODELS_FILE)
    pool = KeyPool.from_settings(settings)

    fl = settings.freellmapi
    if fl.enabled and fl.base_url:  # local FreeLLMAPI gateway, if enabled
        registry.add_provider(
            ProviderDef(
                name="freellmapi",
                base_url=fl.base_url.rstrip("/"),
                daily_quota=fl.daily_quota,
            )
        )
        registry.add_model(
            ModelEntry(
                id="freellmapi-auto",
                provider="freellmapi",
                model=fl.model_chat or "auto",
                tier=2,
                priority=fl.priority,
            )
        )
        pool.add("freellmapi", fl.api_key or "local")

    unknown = pool.providers() - registry.provider_names()
    for p in sorted(unknown):
        log.warning("keys_for_unknown_provider provider=%s (add it to models.json)", p)

    usable = [
        e for e in registry.models if e.enabled and e.provider in pool.providers()
    ]
    if not usable:
        raise RegistryError(
            "no usable models: none of your keys match a provider in models.json"
        )
    engine = PoolEngine(settings, registry, pool, quota, client_factory)
    log.info("pool_engine_ready %s", engine.summary())
    return engine
