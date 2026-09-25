"""
Short-term conversation memory (in-RAM, per-user, bounded).
"""
from __future__ import annotations

from collections import defaultdict, deque

from app.services.ai.models import ChatMessage
from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_HISTORY_MESSAGES = 30


class MemoryStore:
    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        self.max_messages = max_messages
        self._store: dict[int, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )

    def get(self, user_id: int) -> list[ChatMessage]:
        return list(self._store[user_id])

    def add_user(self, user_id: int, text: str) -> None:
        self._store[user_id].append(ChatMessage(role="user", content=text))

    def add_assistant(self, user_id: int, text: str) -> None:
        self._store[user_id].append(ChatMessage(role="assistant", content=text))

    def clear(self, user_id: int) -> None:
        self._store.pop(user_id, None)
        log.info("memory_cleared user_id=%s", user_id)

    def size(self, user_id: int) -> int:
        return len(self._store[user_id])
