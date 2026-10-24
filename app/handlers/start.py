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
/memory — Show what I remember about you
/remember <text> — Save a fact permanently
/forget <number|all> — Delete a saved fact
/reset — Clear the conversation (saved facts stay)
/status — Show AI provider status

You can also just send a normal message — it will be treated as /chat.
Tip: say "yaad rakhna ki ..." and I will remember it.
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
    uid = message.from_user.id
    await memory.ensure_loaded(uid)
    lines = [f"🧠 Memory holds {memory.size(uid)} recent messages."]
    if memory.summary(uid):
        lines.append("📝 I also keep a summary of your older conversation.")
    facts = memory.facts(uid)
    if facts:
        lines.append("\nSaved facts:")
        lines += [f"{i}. {f}" for i, f in enumerate(facts, 1)]
    else:
        lines.append("No saved facts. Use /remember <text>.")
    await message.answer("\n".join(lines), parse_mode=None)


@router.message(Command("remember"))
async def cmd_remember(message: Message, memory: MemoryStore) -> None:
    if not message.from_user:
        return
    uid = message.from_user.id
    fact = (message.text or "").partition(" ")[2].strip()
    if not fact:
        await message.answer("Send: /remember your fact here", parse_mode=None)
        return
    await memory.ensure_loaded(uid)
    if memory.add_fact(uid, fact):
        await message.answer("✅ Saved. I will remember this.", parse_mode=None)
    else:
        await message.answer("Already saved.", parse_mode=None)


@router.message(Command("forget"))
async def cmd_forget(message: Message, memory: MemoryStore) -> None:
    if not message.from_user:
        return
    uid = message.from_user.id
    arg = (message.text or "").partition(" ")[2].strip().lower()
    await memory.ensure_loaded(uid)
    if arg == "all":
        n = memory.clear_facts(uid)
        await message.answer(f"🗑 Deleted {n} saved facts.", parse_mode=None)
    elif arg.isdigit():
        removed = memory.remove_fact(uid, int(arg))
        text = f"🗑 Deleted: {removed}" if removed else "No fact with that number."
        await message.answer(text, parse_mode=None)
    else:
        await message.answer("Send: /forget <number> or /forget all (see /memory)", parse_mode=None)


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
    summary = getattr(agent_router.engine, "summary", None)
    if callable(summary):
        info = summary()
        lines.append(f"\n🔑 {info['keys']} keys · 🤖 {info['models']} models")
    await message.answer("\n".join(lines), parse_mode="Markdown")
