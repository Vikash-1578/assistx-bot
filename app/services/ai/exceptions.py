"""Provider exception hierarchy."""
from __future__ import annotations


class AIError(Exception):
    """Base for all AI-related errors."""


class ProviderError(AIError):
    """Generic provider failure."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class RateLimitError(ProviderError):
    """HTTP 429 or equivalent."""

    def __init__(
        self, provider: str, message: str, retry_after: float | None = None
    ) -> None:
        super().__init__(provider, message, retryable=True)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """401/403 — do not retry."""

    def __init__(self, provider: str, message: str = "authentication failed") -> None:
        super().__init__(provider, message, retryable=False)


class InvalidRequestError(ProviderError):
    """400/422 — malformed request, do not retry."""

    def __init__(self, provider: str, message: str = "invalid request") -> None:
        super().__init__(provider, message, retryable=False)


class ModelNotFoundError(ProviderError):
    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, f"model not found: {model}", retryable=False)
        self.model = model


class TimeoutError_(ProviderError):
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "request timed out", retryable=True)


class ServerError(ProviderError):
    """5xx — retryable."""

    def __init__(self, provider: str, status: int) -> None:
        super().__init__(provider, f"server error {status}", retryable=True)
        self.status = status


class QuotaExhausted(AIError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"[{provider}] daily quota exhausted")
        self.provider = provider


class AllProvidersFailed(AIError):
    """Raised when every candidate failed."""

    def __init__(self, attempted: list[str]) -> None:
        super().__init__(f"all providers failed: {', '.join(attempted)}")
        self.attempted = attempted
