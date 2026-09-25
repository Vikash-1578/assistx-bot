"""FreeLLMAPI — local gateway provider (OpenAI-compatible)."""
from __future__ import annotations

from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider


class FreeLLMAPIProvider(OpenAICompatibleProvider):
    """
    Connects to the FreeLLMAPI gateway running inside the same container
    (localhost:3001). Gateway routes requests across 34+ free LLM providers.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 45) -> None:
        super().__init__(
            name="freellmapi",
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
