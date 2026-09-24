"""
Reusable OpenAI-compatible chat-completions client.
Used by Groq, OpenRouter, Cerebras, SambaNova, HF, Gemini (openai mode).
"""
from __future__ import annotations

import time
from typing import Any

import aiohttp

from app.services.ai.exceptions import (
    AuthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError_,
)
from app.services.ai.models import AIResponse, ChatMessage, TaskType
from app.services.ai.providers.base import BaseProvider
from app.utils.logging import get_logger

log = get_logger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        timeout: int = 45,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout)
        self.name = name
        self._session: aiohttp.ClientSession | None = None
        self._extra_headers = extra_headers or {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=10,
                sock_read=self.timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        h.update(self._extra_headers)
        return h

    async def _post_chat(self, payload: dict) -> tuple[dict, float]:
        session = await self._get_session()
        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload, headers=self._headers()) as resp:
                text = await resp.text()
                latency = (time.monotonic() - t0) * 1000.0
                if resp.status == 200:
                    try:
                        return await resp.json(), latency
                    except Exception as e:
                        raise ProviderError(
                            self.name, f"bad json: {e}", retryable=False
                        )
                self._raise_for_status(resp, text)
                raise ProviderError(
                    self.name, f"unreachable status={resp.status}"
                )
        except aiohttp.ClientConnectorError as e:
            raise ProviderError(self.name, f"connection error: {e}", retryable=True)
        except aiohttp.ServerTimeoutError:
            raise TimeoutError_(self.name)
        except TimeoutError:
            raise TimeoutError_(self.name)

    def _raise_for_status(self, resp: aiohttp.ClientResponse, body: str) -> None:
        status = resp.status
        body_preview = (body or "")[:300]
        if status == 401 or status == 403:
            raise AuthError(self.name, f"http {status}")
        if status == 404:
            raise ModelNotFoundError(self.name, "unknown")
        if status == 429:
            ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            try:
                ra_f = float(ra) if ra is not None else None
            except ValueError:
                ra_f = None
            raise RateLimitError(self.name, "rate limited", retry_after=ra_f)
        if status == 400 or status == 422:
            raise InvalidRequestError(self.name, f"http {status}: {body_preview}")
        if 500 <= status < 600:
            raise ServerError(self.name, status)
        raise ProviderError(
            self.name, f"http {status}: {body_preview}", retryable=False
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        task: TaskType,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data, latency = await self._post_chat(payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                self.name, "no choices in response", retryable=False
            )
        content = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return AIResponse(
            text=content.strip(),
            provider=self.name,
            model=model,
            attempts=1,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            latency_ms=int(latency),
        )
