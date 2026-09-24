"""
AgentRouter — maps user intent to TaskType + system prompt + AI call.

This is the single bridge between Telegram handlers and the AIEngine.
Handlers should never touch providers or prompts directly.
"""
from __future__ import annotations

from pathlib import Path

from app.services.ai.engine import AIEngine
from app.services.ai.exceptions import AllProvidersFailed
from app.services.ai.models import AIResponse, ChatMessage, TaskType
from app.utils.logging import get_logger

log = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PROMPT_FILES: dict[TaskType, str] = {
    TaskType.CHAT: "chat_system.txt",
    TaskType.CODE: "coding_system.txt",
    TaskType.CONTENT: "content_system.txt",
    TaskType.FREELANCE: "freelance_system.txt",
    TaskType.SUMMARY: "summary_system.txt",
    TaskType.REASONING: "reasoning_system.txt",
}

_PROMPT_CACHE: dict[TaskType, str] = {}


def load_prompt(task: TaskType) -> str:
    """Load and cache system prompt for the given task."""
    if task in _PROMPT_CACHE:
        return _PROMPT_CACHE[task]
    filename = _PROMPT_FILES.get(task)
    if not filename:
        return "You are a helpful AI assistant."
    path = _PROMPTS_DIR / filename
    if not path.exists():
        log.warning("prompt_file_missing task=%s path=%s", task.value, path)
        return "You are a helpful AI assistant."
    text = path.read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[task] = text
    return text


class AgentRouter:
    def __init__(self, engine: AIEngine) -> None:
        self.engine = engine

    async def run(
        self,
        user_text: str,
        *,
        task: TaskType = TaskType.CHAT,
        history: list[ChatMessage] | None = None,
        user_id: int | None = None,
        extra_system: str | None = None,
    ) -> AIResponse:
        """
        Build the message list and delegate to AIEngine.

        history: previous turns (short-term), already in chronological order.
        """
        system_prompt = load_prompt(task)
        if extra_system:
            system_prompt = f"{system_prompt}\n\n{extra_system}"

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_text))

        try:
            return await self.engine.generate(
                messages, task=task, user_id=user_id
            )
        except AllProvidersFailed as e:
            log.error("all_providers_failed attempted=%s", e.attempted)
            raise
