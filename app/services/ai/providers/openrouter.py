from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, base_url: str, timeout: int = 45) -> None:
        super().__init__(
            name="openrouter",
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            extra_headers={
                "HTTP-Referer": "https://github.com/telegram-ai-agent",
                "X-Title": "Telegram AI Agent",
            },
        )
