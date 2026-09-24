"""Content handler — /content command."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.handlers.chat import _run_ai
from app.services.agent_router import AgentRouter
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore

router = Router(name="content")


@router.message(Command("content"))
async def cmd_content(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer(
            "Send: /content what you want to create.\n"
            "Example:\n/content YouTube script on Python tips"
        )
        return
    await _run_ai(message, text, TaskType.CONTENT, agent_router, memory)
