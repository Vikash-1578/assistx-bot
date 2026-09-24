"""Provider abstract base."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.ai.models import AIResponse, ChatMessage, TaskType


class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, api_key: str, base_url: str, timeout: int = 45) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        task: TaskType,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse: ...

    @abstractmethod
    async def close(self) -> None: ...
