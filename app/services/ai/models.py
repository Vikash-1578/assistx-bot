"""Normalized AI data models — provider-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    CONTENT = "content"
    FREELANCE = "freelance"
    REASONING = "reasoning"
    SUMMARY = "summary"


Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


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
class ProviderHealth:
    name: str
    enabled: bool
    healthy: bool
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    requests_today: int = 0
    daily_limit: int = 0
    remaining: int = 0


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
