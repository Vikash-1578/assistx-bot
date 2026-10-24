"""
AgentRouter — maps user intent to TaskType + system prompt + AI call, and
adds the "smart" layer on top of whatever engine is behind it:

  * language-matched, policy-guided system prompt
  * automatic task detection for plain chat (auto_task=True)
  * model tiering by question difficulty (when the engine supports it)
  * output cleaning (<think> blocks etc.)
  * cut-off answers are continued instead of ending mid-sentence
  * answers that fail the quality gate are retried on a stronger model
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.services.ai.exceptions import AllProvidersFailed
from app.services.ai.models import AIResponse, ChatMessage, TaskType
from app.services.smart import (
    LANGUAGE_HINTS,
    PREFER_BY_DIFFICULTY,
    classify,
    clean_output,
    has_unclosed_fence,
    merge_continuation,
    quality_check,
)
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from app.services.ai.engine import AIEngine

log = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PROMPT_FILES: dict[TaskType, str] = {
    TaskType.CHAT: "chat_system.txt",
    TaskType.CODE: "coding_system.txt",
    TaskType.CONTENT: "content_system.txt",
    TaskType.FREELANCE: "freelance_system.txt",
    TaskType.SUMMARY: "summary_system.txt",
    TaskType.REASONING: "reasoning_system.txt",
    TaskType.VISION: "vision_system.txt",
}

_PROMPT_CACHE: dict[TaskType, str] = {}
_POLICY_CACHE: list[str] = []

_FALLBACK_POLICY = (
    "Answer directly and correctly. Never invent facts. Match the user's "
    "language. Keep formatting simple."
)
_CONTINUE_PROMPT = (
    "Your last message was cut off. Continue exactly where you stopped. "
    "Do not repeat anything you already wrote."
)


def load_prompt(task: TaskType) -> str:
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


def load_policy() -> str:
    if _POLICY_CACHE:
        return _POLICY_CACHE[0]
    path = _PROMPTS_DIR / "answer_policy.txt"
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    _POLICY_CACHE.append(text or _FALLBACK_POLICY)
    return _POLICY_CACHE[0]


class AgentRouter:
    def __init__(
        self,
        engine: "AIEngine | Any",
        *,
        smart_enabled: bool = True,
        max_escalations: int = 2,
    ) -> None:
        self.engine = engine
        self.smart_enabled = smart_enabled
        self.max_escalations = max(0, max_escalations)

    # ------------------------------------------------------------- helpers
    @property
    def _tiers(self) -> bool:
        """True if the engine can pick models by tier (PoolEngine)."""
        return bool(getattr(self.engine, "supports_tiers", False))

    async def _call(
        self,
        messages: list[ChatMessage],
        task: TaskType,
        user_id: int | None,
        prefer: str,
        exclude: set[str],
    ) -> AIResponse:
        if self._tiers:
            return await self.engine.generate(
                messages,
                task=task,
                user_id=user_id,
                prefer=prefer,
                exclude=set(exclude),
            )
        return await self.engine.generate(messages, task=task, user_id=user_id)

    @staticmethod
    def _system_prompt(task: TaskType, language: str, extra: str | None) -> str:
        parts = [
            load_prompt(task),
            load_policy(),
            LANGUAGE_HINTS.get(language, LANGUAGE_HINTS["other"]),
            extra or "",
        ]
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _with_hint(messages: list[ChatMessage], issues: list[str]) -> list[ChatMessage]:
        hint = (
            "\n\nNOTE: a previous attempt at this answer was rejected for: "
            + ", ".join(issues)
            + ". Answer again, carefully and completely, in the user's language."
        )
        first = messages[0]
        return [ChatMessage(role=first.role, content=f"{first.content}{hint}")] + list(
            messages[1:]
        )

    async def _polish(
        self,
        resp: AIResponse,
        messages: list[ChatMessage],
        task: TaskType,
        user_id: int | None,
        prefer: str,
        exclude: set[str],
    ) -> AIResponse:
        """Clean the text; continue it if the model was cut off."""
        text = clean_output(resp.text)
        if resp.finish_reason == "length" or has_unclosed_fence(text):
            try:
                more = await self._call(
                    list(messages)
                    + [
                        ChatMessage(role="assistant", content=text),
                        ChatMessage(role="user", content=_CONTINUE_PROMPT),
                    ],
                    task,
                    user_id,
                    prefer,
                    exclude,
                )
                text = merge_continuation(text, clean_output(more.text))
            except AllProvidersFailed:
                log.warning("continuation_failed user=%s", user_id)
            if has_unclosed_fence(text):
                text += "\n```"
        resp.text = text
        return resp

    async def _generate_smart(
        self,
        messages: list[ChatMessage],
        user_text: str,
        task: TaskType,
        language: str,
        user_id: int | None,
        prefer: str,
    ) -> AIResponse:
        if not self.smart_enabled:
            return await self._call(messages, task, user_id, prefer, set())

        exclude: set[str] = set()
        best: AIResponse | None = None
        best_issues = 10**6
        msgs = messages
        attempts = self.max_escalations + 1 if self._tiers else 1
        for i in range(attempts):
            try:
                resp = await self._call(msgs, task, user_id, prefer, exclude)
            except AllProvidersFailed:
                if best is not None:  # nothing stronger left; keep what we have
                    break
                raise
            resp = await self._polish(resp, msgs, task, user_id, prefer, exclude)
            gate = quality_check(user_text, resp.text, task=task, language=language)
            if gate.ok:
                return resp
            log.warning(
                "quality_gate_failed model=%s issues=%s attempt=%s",
                resp.entry_id or resp.model, gate.issues, i + 1,
            )
            if len(gate.issues) < best_issues:
                best, best_issues = resp, len(gate.issues)
            if resp.entry_id:
                exclude.add(resp.entry_id)
            prefer = "quality"
            msgs = self._with_hint(messages, gate.issues)
        assert best is not None
        return best

    # --------------------------------------------------------------- public
    async def run(
        self,
        user_text: str,
        *,
        task: TaskType = TaskType.CHAT,
        history: list[ChatMessage] | None = None,
        user_id: int | None = None,
        extra_system: str | None = None,
        images_b64: list[str] | None = None,
        image_mime: str = "image/jpeg",
        auto_task: bool = False,
    ) -> AIResponse:
        cls = classify(user_text, has_image=bool(images_b64))
        if auto_task and task == TaskType.CHAT and not images_b64:
            task = cls.task

        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=self._system_prompt(task, cls.language, extra_system),
            )
        ]
        if history:
            messages.extend(history)

        if images_b64:
            messages.append(
                ChatMessage.with_images("user", user_text, images_b64, image_mime)
            )
        else:
            messages.append(ChatMessage(role="user", content=user_text))

        prefer = PREFER_BY_DIFFICULTY.get(cls.difficulty, "balanced")
        try:
            return await self._generate_smart(
                messages, user_text, task, cls.language, user_id, prefer
            )
        except AllProvidersFailed as e:
            log.error("all_providers_failed attempted=%s", e.attempted)
            raise

    async def summarize_history(
        self, previous: str, messages: list[ChatMessage]
    ) -> str:
        """Fold old chat messages into a short running summary (for memory)."""

        def line(m: ChatMessage) -> str:
            body = m.content if isinstance(m.content, str) else "[media]"
            return f"{m.role}: {body}"[:600]

        transcript = "\n".join(line(m) for m in messages)
        prompt = (
            "Update the running summary of a chat between a user and an AI "
            "assistant.\n\n"
            f"Previous summary:\n{previous or '(none)'}\n\n"
            f"New messages:\n{transcript}\n\n"
            "Write the updated summary in under 180 words. Keep: the user's name "
            "and facts about them, their goals and ongoing projects, preferences, "
            "decisions made and open tasks. Drop small talk. Use the language "
            "the user writes in. Output only the summary."
        )
        resp = await self._call(
            [
                ChatMessage(role="system", content=load_prompt(TaskType.SUMMARY)),
                ChatMessage(role="user", content=prompt),
            ],
            TaskType.SUMMARY,
            None,
            "speed",
            set(),
        )
        return clean_output(resp.text)[:1500]
