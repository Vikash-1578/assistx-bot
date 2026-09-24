"""Chat handler — normal text + /chat command."""
from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from app.services.agent_router import AgentRouter
from app.services.ai.exceptions import AllProvidersFailed
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore
from app.utils.logging import get_logger
from app.utils.security import sanitize_user_text

log = get_logger(__name__)
router = Router(name="chat")


async def _run_ai(
    message: Message,
    text: str,
    task: TaskType,
    agent_router: AgentRouter,
    memory: MemoryStore,
    use_history: bool = True,
) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id

    text = sanitize_user_text(text)
    if not text:
        await message.answer("⚠ Please send some text.")
        return

    placeholder = await message.answer("⏳ Processing...")

    try:
        history = memory.get(user_id) if use_history else None
        response = await agent_router.run(
            text,
            task=task,
            history=history,
            user_id=user_id,
        )
        memory.add_user(user_id, text)
        memory.add_assistant(user_id, response.text)

        reply = response.text or "(empty response)"
        if len(reply) > 4000:
            # Telegram limit ~4096; chunk if needed
            for i in range(0, len(reply), 4000):
                await message.answer(reply[i : i + 4000])
        else:
            await message.answer(reply)

        try:
            await placeholder.delete()
        except Exception:
            pass

    except AllProvidersFailed:
        log.error("chat_all_providers_failed user_id=%s", user_id)
        try:
            await placeholder.edit_text(
                "⚠ All AI providers are temporarily unavailable. "
                "Please try again in a moment."
            )
        except Exception:
            pass
    except Exception as e:
        log.exception("chat_handler_error user_id=%s err=%s", user_id, e)
        try:
            await placeholder.edit_text(
                "❌ Something went wrong while processing your request. "
                "Please try again."
            )
        except Exception:
            pass


@router.message(Command("chat"))
async def cmd_chat(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer("Send: /chat your message here")
        return
    await _run_ai(message, text, TaskType.CHAT, agent_router, memory)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    """Any normal text message → CHAT mode."""
    await _run_ai(
        message,
        message.text or "",
        TaskType.CHAT,
        agent_router,
        memory,
    )
