"""Factory that wires provider clients from settings."""
from __future__ import annotations

from app.config import Settings
from app.services.ai.providers.base import BaseProvider
from app.services.ai.providers.cerebras import CerebrasProvider
from app.services.ai.providers.gemini import GeminiProvider
from app.services.ai.providers.groq import GroqProvider
from app.services.ai.providers.huggingface import HuggingFaceProvider
from app.services.ai.providers.nvidia import NvidiaProvider
from app.services.ai.providers.freellmapi import FreeLLMAPIProvider
from app.services.ai.providers.openrouter import OpenRouterProvider
from app.services.ai.providers.sambanova import SambaNovaProvider
from app.utils.logging import get_logger

log = get_logger(__name__)


def build_providers(settings: Settings) -> dict[str, BaseProvider]:
    """Build provider clients for all enabled providers."""
    timeout = settings.ai.request_timeout
    providers: dict[str, BaseProvider] = {}

    if settings.groq.enabled and settings.groq.api_key:
        providers["groq"] = GroqProvider(
            settings.groq.api_key, settings.groq.base_url, timeout
        )

    if settings.gemini.enabled and settings.gemini.api_key:
        providers["gemini"] = GeminiProvider(
            settings.gemini.api_key, settings.gemini.base_url, timeout
        )

    if settings.openrouter.enabled and settings.openrouter.api_key:
        providers["openrouter"] = OpenRouterProvider(
            settings.openrouter.api_key, settings.openrouter.base_url, timeout
        )

    if settings.cerebras.enabled and settings.cerebras.api_key:
        providers["cerebras"] = CerebrasProvider(
            settings.cerebras.api_key, settings.cerebras.base_url, timeout
        )

    if settings.sambanova.enabled and settings.sambanova.api_key:
        providers["sambanova"] = SambaNovaProvider(
            settings.sambanova.api_key, settings.sambanova.base_url, timeout
        )

    if settings.huggingface.enabled and settings.huggingface.api_key:
        providers["huggingface"] = HuggingFaceProvider(
            settings.huggingface.api_key,
            settings.huggingface.base_url,
            timeout,
        )

    if settings.nvidia.enabled and settings.nvidia.api_key:
        providers["nvidia"] = NvidiaProvider(
            settings.nvidia.api_key,
            settings.nvidia.base_url,
            timeout,
        )

    if settings.freellmapi.enabled and settings.freellmapi.base_url:
        providers["freellmapi"] = FreeLLMAPIProvider(
            settings.freellmapi.api_key or "local",
            settings.freellmapi.base_url,
            timeout,
        )

    log.info("providers_built names=%s", list(providers.keys()))
    return providers
