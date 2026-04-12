import asyncio
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import *

# ═══ LOGGING ═══
logging.basicConfig(level=logging.INFO)

# ═══ BOT SETUP ═══
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ═══ STATES ═══
class ChatState(StatesGroup):
    waiting_for_message = State()

# ═══ MAIN MENU ═══
def main_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🧠 Chat", callback_data="chat"),
            InlineKeyboardButton(text="🎨 Create", callback_data="create")
        ],
        [
            InlineKeyboardButton(text="🔍 Research", callback_data="research"),
            InlineKeyboardButton(text="📁 Upload", callback_data="upload")
        ],
        [
            InlineKeyboardButton(text="📚 Learn", callback_data="learn"),
            InlineKeyboardButton(text="🛠 Tools", callback_data="tools")
        ],
        [
            InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
            InlineKeyboardButton(text="💎 Plans", callback_data="plans")
        ]
    ])

# ═══ CHAT MODES MENU ═══
def chat_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⚡ Fast", callback_data="mode_fast"),
            InlineKeyboardButton(text="⚖️ Balanced", callback_data="mode_balanced")
        ],
        [
            InlineKeyboardButton(text="🧠 Smart", callback_data="mode_smart"),
            InlineKeyboardButton(text="🔬 Deep", callback_data="mode_deep")
        ],
        [
            InlineKeyboardButton(text="✨ Creative", callback_data="mode_creative"),
            InlineKeyboardButton(text="💻 Coding", callback_data="mode_coding")
        ],
        [
            InlineKeyboardButton(text="🎓 Study", callback_data="mode_study")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="back_main")
        ]
    ])

# ═══ BACK BUTTON ═══
def back_button(callback):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔙 Back", callback_data=callback)]
    ])

# ═══ CHAT ACTIVE MENU ═══
def chat_active_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔄 Change Mode", callback_data="chat"),
            InlineKeyboardButton(text="🗑 Clear History", callback_data="clear_history")
        ],
        [
            InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_main")
        ]
    ])

# ═══ WELCOME ═══
async def send_welcome(message: Message):
    name = message.from_user.first_name
    await message.answer(
        f"⚡ *Welcome to AssistX, {name}!*\n\n"
        f"Your all-in-one AI assistant.\n\n"
        f"🧠 Chat with AI\n"
        f"🎨 Generate Images\n"
        f"🔍 Search the Web\n"
        f"📄 Read your Files\n"
        f"📚 Learn Anything\n\n"
        f"*What would you like to do?* 👇",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )
# ═══ /start HANDLER ═══
@dp.message(F.text == "/start")
async def start_handler(message: Message, state: FSMContext):
    await state.clear()
    await send_welcome(message)

# ═══ MAIN MENU BUTTONS ═══
@dp.callback_query(F.data == "back_main")
async def back_main(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        f"⚡ *AssistX — Main Menu*\n\n"
        f"*What would you like to do?* 👇",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )

# ═══ CHAT BUTTON ═══
@dp.callback_query(F.data == "chat")
async def chat_handler(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🧠 *Chat Mode*\n\n"
        f"Choose how you want AI to respond:\n\n"
        f"⚡ *Fast* — Quick short answers\n"
        f"⚖️ *Balanced* — Normal helpful answers\n"
        f"🧠 *Smart* — Better reasoning\n"
        f"🔬 *Deep* — Detailed analysis\n"
        f"✨ *Creative* — Ideas & writing\n"
        f"💻 *Coding* — Code & debugging\n"
        f"🎓 *Study* — Student friendly\n",
        parse_mode="Markdown",
        reply_markup=chat_menu()
    )

# ═══ CHAT MODE SELECTED ═══
@dp.callback_query(F.data.startswith("mode_"))
async def mode_selected(callback: CallbackQuery, state: FSMContext):
    mode = callback.data.replace("mode_", "")

    mode_info = {
        "fast":     ("⚡", "Fast Mode",     "Quick and concise answers."),
        "balanced": ("⚖️", "Balanced Mode", "Helpful and clear answers."),
        "smart":    ("🧠", "Smart Mode",    "Better reasoning and explanations."),
        "deep":     ("🔬", "Deep Mode",     "Detailed analysis and research."),
        "creative": ("✨", "Creative Mode", "Ideas, writing and brainstorming."),
        "coding":   ("💻", "Coding Mode",   "Code generation and debugging."),
        "study":    ("🎓", "Study Mode",    "Simple student-friendly explanations.")
    }

    emoji, name, desc = mode_info[mode]

    await state.update_data(mode=mode)
    await state.set_state(ChatState.waiting_for_message)

    await callback.message.edit_text(
        f"{emoji} *{name} Activated!*\n\n"
        f"{desc}\n\n"
        f"*Send your message now* 👇\n\n"
        f"_Tip: You can change mode or clear history anytime._",
        parse_mode="Markdown",
        reply_markup=chat_active_menu()
    )

# ═══ CLEAR HISTORY ═══
@dp.callback_query(F.data == "clear_history")
async def clear_history(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    mode = data.get("mode", "balanced")
    await state.update_data(history=[], mode=mode)
    await callback.answer("✅ History cleared!", show_alert=False)

# ═══ COMING SOON BUTTONS ═══
@dp.callback_query(F.data.in_({"create", "research", "upload", "learn", "tools", "settings", "plans"}))
async def coming_soon(callback: CallbackQuery):
    names = {
        "create":   "🎨 Create Studio",
        "research": "🔍 Research",
        "upload":   "📁 Upload",
        "learn":    "📚 Learn",
        "tools":    "🛠 Tools",
        "settings": "⚙️ Settings",
        "plans":    "💎 Plans"
    }
    name = names[callback.data]
    await callback.message.edit_text(
        f"{name}\n\n"
        f"🚧 *Coming Soon!*\n\n"
        f"This feature is under development.\n"
        f"Stay tuned for updates! 🚀",
        parse_mode="Markdown",
        reply_markup=back_button("back_main")
    )

# ═══ AI RESPONSE ═══
@dp.message(ChatState.waiting_for_message)
async def ai_response(message: Message, state: FSMContext):
    data = await state.get_data()
    mode = data.get("mode", "balanced")
    history = data.get("history", [])

    # Typing indicator
    await bot.send_chat_action(message.chat.id, "typing")

    # System prompts per mode
    system_prompts = {
        "fast":     "You are a fast assistant. Give very short, direct answers. Max 3 sentences.",
        "balanced": "You are a helpful assistant. Give clear and useful answers.",
        "smart":    "You are a smart assistant. Give well-reasoned, detailed explanations.",
        "deep":     "You are a deep research assistant. Analyze thoroughly with examples.",
        "creative": "You are a creative assistant. Give imaginative, inspiring responses.",
        "coding":   "You are a coding expert. Give clean code with brief explanations.",
        "study":    "You are a study tutor. Explain simply like teaching a student."
    }

    system = system_prompts.get(mode, system_prompts["balanced"])

    # Add user message to history
    history.append({"role": "user", "content": message.text})

    # Keep history limit
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}] + history,
            max_tokens=1024
        )
        reply = response.choices[0].message.content

    except Exception as e:
        reply = (
            "⚠️ *AI is temporarily unavailable.*\n\n"
            "Please try again in a moment."
        )

    # Add assistant reply to history
    history.append({"role": "assistant", "content": reply})
    await state.update_data(history=history)

    await message.answer(
        reply,
        parse_mode="Markdown",
        reply_markup=chat_active_menu()
    )

# ═══ RUN BOT ═══
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())