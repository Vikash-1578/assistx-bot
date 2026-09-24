"""Freelance handler — /freelance command + FSM proposal workflow."""
from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from app.handlers.chat import _run_ai
from app.services.agent_router import AgentRouter
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore
from app.states.freelance import FreelanceProposal

router = Router(name="freelance")


@router.message(Command("freelance"))
async def cmd_freelance(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
    state: FSMContext,
) -> None:
    text = (message.text or "").partition(" ")[2].strip()

    # No argument → start proposal workflow
    if not text:
        await state.set_state(FreelanceProposal.waiting_client_name)
        await message.answer(
            "📝 *Freelance Proposal Builder*\n\n"
            "I'll ask you a few quick questions.\n"
            "Send /cancel anytime to stop.\n\n"
            "1️⃣ What is the *client's name*?",
            parse_mode="Markdown",
        )
        return

    # With argument → one-shot freelance assistance
    await _run_ai(message, text, TaskType.FREELANCE, agent_router, memory)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Nothing to cancel.")
        return
    await state.clear()
    await message.answer("❌ Cancelled.")


@router.message(FreelanceProposal.waiting_client_name)
async def step_client(message: Message, state: FSMContext) -> None:
    await state.update_data(client_name=(message.text or "").strip())
    await state.set_state(FreelanceProposal.waiting_project_desc)
    await message.answer("2️⃣ What is the *project description*? (1-3 lines)")


@router.message(FreelanceProposal.waiting_project_desc)
async def step_desc(message: Message, state: FSMContext) -> None:
    await state.update_data(project_desc=(message.text or "").strip())
    await state.set_state(FreelanceProposal.waiting_budget)
    await message.answer("3️⃣ What is the *budget*? (e.g. $500 or ₹40000)")


@router.message(FreelanceProposal.waiting_budget)
async def step_budget(message: Message, state: FSMContext) -> None:
    await state.update_data(budget=(message.text or "").strip())
    await state.set_state(FreelanceProposal.waiting_deadline)
    await message.answer("4️⃣ What is the *deadline*? (e.g. 2 weeks, 30 Oct)")


@router.message(FreelanceProposal.waiting_deadline)
async def step_deadline(message: Message, state: FSMContext) -> None:
    await state.update_data(deadline=(message.text or "").strip())
    await state.set_state(FreelanceProposal.waiting_tone)
    await message.answer(
        "5️⃣ Preferred *tone*? (formal / friendly / persuasive — or just say 'any')"
    )


@router.message(FreelanceProposal.waiting_tone)
async def step_tone(
    message: Message,
    state: FSMContext,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    tone = (message.text or "").strip()
    data = await state.get_data()
    await state.clear()

    if not message.from_user:
        return

    prompt = (
        "Draft a professional freelance proposal.\n\n"
        f"Client: {data.get('client_name', 'N/A')}\n"
        f"Project: {data.get('project_desc', 'N/A')}\n"
        f"Budget: {data.get('budget', 'N/A')}\n"
        f"Deadline: {data.get('deadline', 'N/A')}\n"
        f"Tone: {tone}\n\n"
        "Output the proposal in a clean, ready-to-send format."
    )

    await _run_ai(
        message,
        prompt,
        TaskType.FREELANCE,
        agent_router,
        memory,
        use_history=False,
    )
