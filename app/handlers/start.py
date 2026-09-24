"""Basic commands: /start /help /reset /memory /status"""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.services.agent_router import AgentRouter
from app.services.memory import MemoryStore
from app.utils.logging import get_logger

log = get_logger(__name__)
router = Router(name="start")


HELP_TEXT = """🤖 *Telegram AI Agent*

Available commands:
/start — Welcome message
/help — Show this help
/chat — Normal conversation (default)
/code — Coding assistant (debug, refactor, explain)
/freelance — Freelance helper (proposals, client msgs)
/content — Content creation (YouTube, blog, social)
/summary — Summarization
/memory — Show memory size
/reset — Clear your conversation memory
/status — Show AI provider status

You can also just send a normal message — it will be treated as /chat.
"""


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    name = message.from_user.first_name if message.from_user else "there"
    await message.answer(
        f"👋 Hello {name}!\n\n"
        "I'm your personal AI assistant. Send me any message, or use /help "
        "to see available commands."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, parse_mode="Markdown")


@router.message(Command("reset"))
async def cmd_reset(message: Message, memory: MemoryStore) -> None:
    if message.from_user:
        memory.clear(message.from_user.id)
    await message.answer("🧹 Conversation memory cleared.")


@router.message(Command("memory"))
async def cmd_memory(message: Message, memory: MemoryStore) -> None:
    if not message.from_user:
        return
    size = memory.size(message.from_user.id)
    await message.answer(f"🧠 Memory holds {size} recent messages.")


@router.message(Command("status"))
async def cmd_status(message: Message, agent_router: AgentRouter) -> None:
    statuses = await agent_router.engine.status()
    lines = ["📊 *AI Engine Status*\n"]
    for s in statuses:
        if not s.enabled:
            mark = "○ Disabled"
        elif s.available:
            mark = "✓ Available"
        elif s.cooldown_seconds_left > 0:
            mark = f"⏸ Cooldown {s.cooldown_seconds_left}s"
        elif s.requests_today >= s.daily_limit:
            mark = "⚠ Quota exhausted"
        else:
            mark = "✗ Unavailable"
        lines.append(
            f"*{s.name.capitalize()}*\n"
            f"  {mark}\n"
            f"  Requests today: {s.requests_today} / {s.daily_limit}"
        )
    await message.answer("\n".join(lines), parse_mode="Markdown")
