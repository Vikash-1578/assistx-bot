"""NVIDIA NIM — OpenAI-compatible endpoint."""
from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, base_url: str, timeout: int = 45) -> None:
        super().__init__(name="nvidia", api_key=api_key, base_url=base_url, timeout=timeout)
