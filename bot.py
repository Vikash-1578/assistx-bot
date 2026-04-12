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