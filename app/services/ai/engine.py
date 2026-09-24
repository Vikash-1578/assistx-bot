"""
AIEngine — the ONLY AI entry point for the rest of the app.

Responsibilities:
- model selection per TaskType
- provider selection via scoring (priority + failures + cooldown + quota)
- retry / fallback
- quota + health tracking
- structured logging

Handlers must call `engine.generate(...)` and nothing else.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.config import AISettings, ProviderSettings, Settings
from app.services.ai.exceptions import (
    AllProvidersFailed,
    AuthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
)
from app.services.ai.models import AIResponse, ChatMessage, ProviderStatus, TaskType
from app.services.ai.providers.base import BaseProvider
from app.services.ai.quota import QuotaTracker
from app.services.ai.retry import with_retry
from app.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _ProviderState:
    name: str
    settings: ProviderSettings
    client: BaseProvider
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""


class AIEngine:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, BaseProvider],
        quota: QuotaTracker,
    ) -> None:
        self.settings = settings
        self.ai_cfg: AISettings = settings.ai
        self._providers: dict[str, _ProviderState] = {}
        provider_settings = settings.enabled_providers()
        for name, client in providers.items():
            pcfg = provider_settings[name]
            self._providers[name] = _ProviderState(
                name=name, settings=pcfg, client=client
            )
        self.quota = quota
        self._global_sem = asyncio.Semaphore(self.ai_cfg.global_concurrency)
        self._user_sems: dict[int, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        for st in self._providers.values():
            try:
                await st.client.close()
            except Exception as e:
                log.warning("provider_close_error provider=%s err=%s", st.name, e)

    def _user_sem(self, user_id: int) -> asyncio.Semaphore:
        sem = self._user_sems.get(user_id)
        if sem is None:
            sem = asyncio.Semaphore(self.ai_cfg.user_concurrency)
            self._user_sems[user_id] = sem
        return sem

    async def _score(self, st: _ProviderState, task: TaskType) -> float:
        now = time.monotonic()
        score = float(st.settings.priority)
        score += st.consecutive_failures * 15.0
        if st.cooldown_until > now:
            score += 10_000.0
        usage = await self.quota.snapshot(st.name, st.settings.daily_quota)
        if usage["remaining"] <= 0:
            score += 10_000.0
        else:
            frac = usage["requests"] / max(1, st.settings.daily_quota)
            score += frac * 50.0
        model = st.settings.model_for(task.value)
        if not model:
            score += 10_000.0
        return score

    async def _select(self, task: TaskType) -> list[_ProviderState]:
        async with self._lock:
            scored: list[tuple[float, _ProviderState]] = []
            for st in self._providers.values():
                if not st.settings.enabled or not st.settings.api_key:
                    continue
                if st.cooldown_until > time.monotonic():
                    continue
                usage = await self.quota.snapshot(st.name, st.settings.daily_quota)
                if usage["remaining"] <= 0:
                    continue
                s = await self._score(st, task)
                scored.append((s, st))
            scored.sort(key=lambda x: x[0])
            return [st for _, st in scored]

    async def _mark_failure(self, st: _ProviderState, err: Exception) -> None:
        async with self._lock:
            st.consecutive_failures += 1
            st.last_error = str(err)
            await self.quota.record_failure(st.name)
            if st.consecutive_failures >= self.ai_cfg.failure_threshold:
                st.cooldown_until = (
                    time.monotonic() + self.ai_cfg.cooldown_seconds
                )
                log.warning(
                    "provider_cooldown provider=%s until=+%ss failures=%s",
                    st.name,
                    self.ai_cfg.cooldown_seconds,
                    st.consecutive_failures,
                )

    async def _mark_success(self, st: _ProviderState) -> None:
        async with self._lock:
            st.consecutive_failures = 0
            st.cooldown_until = 0.0
            st.last_error = ""

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        task: TaskType = TaskType.CHAT,
        user_id: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIResponse:
        temperature = (
            self.ai_cfg.temperature if temperature is None else temperature
        )
        max_tokens = (
            self.ai_cfg.max_output_tokens if max_tokens is None else max_tokens
        )

        candidates = await self._select(task)
        if not candidates:
            raise AllProvidersFailed(attempted=["<none available>"])

        attempted: list[str] = []
        async with self._global_sem:
            user_sem = (
                self._user_sem(user_id) if user_id is not None else None
            )
            if user_sem:
                await user_sem.acquire()
            try:
                for st in candidates:
                    attempted.append(st.name)
                    model = st.settings.model_for(task.value)
                    log.info(
                        "ai_request provider=%s model=%s task=%s user=%s",
                        st.name,
                        model,
                        task.value,
                        user_id,
                    )
                    try:
                        resp = await with_retry(
                            lambda st=st, model=model: st.client.generate(
                                messages,
                                model=model,
                                task=task,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            max_retries=self.ai_cfg.max_retries,
                            provider=st.name,
                        )
                        await self._mark_success(st)
                        await self.quota.record_success(
                            st.name,
                            input_tokens=resp.input_tokens,
                            output_tokens=resp.output_tokens,
                            total_tokens=resp.total_tokens,
                        )
                        log.info(
                            "ai_response provider=%s model=%s latency_ms=%s tokens=%s",
                            st.name,
                            resp.model,
                            resp.latency_ms,
                            resp.total_tokens,
                        )
                        return resp
                    except (
                        AuthError,
                        InvalidRequestError,
                        ModelNotFoundError,
                    ) as e:
                        log.error(
                            "ai_non_retryable provider=%s err=%s", st.name, e
                        )
                        await self._mark_failure(st, e)
                        continue
                    except QuotaExhausted as e:
                        log.warning("ai_quota_exhausted provider=%s", st.name)
                        await self._mark_failure(st, e)
                        continue
                    except (
                        RateLimitError,
                        ProviderError,
                        asyncio.TimeoutError,
                    ) as e:
                        log.warning(
                            "ai_provider_failed provider=%s err=%s", st.name, e
                        )
                        await self._mark_failure(st, e)
                        continue
            finally:
                if user_sem:
                    user_sem.release()

        raise AllProvidersFailed(attempted=attempted)

    async def status(self) -> list[ProviderStatus]:
        now = time.monotonic()
        out: list[ProviderStatus] = []
        for st in self._providers.values():
            usage = await self.quota.snapshot(
                st.name, st.settings.daily_quota
            )
            cd_left = (
                max(0, int(st.cooldown_until - now))
                if st.cooldown_until > now
                else 0
            )
            available = (
                st.settings.enabled
                and bool(st.settings.api_key)
                and usage["remaining"] > 0
                and cd_left == 0
            )
            out.append(
                ProviderStatus(
                    name=st.name,
                    enabled=st.settings.enabled,
                    available=available,
                    requests_today=usage["requests"],
                    daily_limit=st.settings.daily_quota,
                    consecutive_failures=st.consecutive_failures,
                    cooldown_seconds_left=cd_left,
                    detail=st.last_error[:120],
                )
            )
        return out
