"""
Gemini via OpenAI-compatible endpoint:
  https://generativelanguage.googleapis.com/v1beta/openai
"""
from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, base_url: str, timeout: int = 45) -> None:
        super().__init__(name="gemini", api_key=api_key, base_url=base_url, timeout=timeout)
