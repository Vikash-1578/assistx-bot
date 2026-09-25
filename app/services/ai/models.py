"""Normalized AI data models — provider-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    CONTENT = "content"
    FREELANCE = "freelance"
    REASONING = "reasoning"
    SUMMARY = "summary"
    VISION = "vision"


Role = Literal["system", "user", "assistant"]

ContentType = str | list[dict[str, Any]]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: ContentType

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def text(cls, role: Role, text: str) -> "ChatMessage":
        return cls(role=role, content=text)

    @classmethod
    def with_images(
        cls,
        role: Role,
        text: str,
        images_b64: list[str],
        mime: str = "image/jpeg",
    ) -> "ChatMessage":
        """Build an OpenAI-compatible multimodal message."""
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for b64 in images_b64:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        return cls(role=role, content=parts)


@dataclass(slots=True)
class AIResponse:
    text: str
    provider: str
    model: str
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


@dataclass(slots=True)
class ProviderStatus:
    name: str
    enabled: bool
    available: bool
    requests_today: int
    daily_limit: int
    consecutive_failures: int
    cooldown_seconds_left: int
    detail: str = ""
