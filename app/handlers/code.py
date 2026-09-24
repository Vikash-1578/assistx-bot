"""Code handler — /code command."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.handlers.chat import _run_ai
from app.services.agent_router import AgentRouter
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore

router = Router(name="code")


@router.message(Command("code"))
async def cmd_code(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer(
            "Send: /code your question or paste the code to debug/refactor.\n"
            "Example:\n/code fix this Python error: IndexError ..."
        )
        return
    await _run_ai(message, text, TaskType.CODE, agent_router, memory)
