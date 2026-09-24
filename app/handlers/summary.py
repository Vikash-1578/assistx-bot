"""Summary handler — /summary command."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.handlers.chat import _run_ai
from app.services.agent_router import AgentRouter
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore

router = Router(name="summary")


@router.message(Command("summary"))
async def cmd_summary(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer(
            "Send: /summary paste text to summarize."
        )
        return
    await _run_ai(
        message, text, TaskType.SUMMARY, agent_router, memory, use_history=False
    )
