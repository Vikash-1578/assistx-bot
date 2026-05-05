# bot.py

import os
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from gtts import gTTS
import edge_tts

# simple temp memory
user_mode = {}
user_engine = {}


# =========================
# START MENU
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎙 Create Voice", callback_data="create_voice")],
        [InlineKeyboardButton("🧠 Select Engine", callback_data="select_engine")],
        [InlineKeyboardButton("📁 My Files", callback_data="my_files")],
        [InlineKeyboardButton("⚙ Settings", callback_data="settings")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Welcome to Ultra Voice AI\n\nPremium Telegram Voice Bot",
        reply_markup=reply_markup
    )


# =========================
# BUTTON HANDLER
# =========================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    uid = query.from_user.id
    data = query.data

    # Create Voice
    if data == "create_voice":
        user_mode[uid] = "await_text"
        await query.message.reply_text(
            "Send your text now.\nI will convert it to voice."
        )

    # Select Engine
    elif data == "select_engine":
        keyboard = [
            [InlineKeyboardButton("Edge TTS", callback_data="engine_edge")],
            [InlineKeyboardButton("gTTS", callback_data="engine_gtts")],
            [InlineKeyboardButton("Google TTS", callback_data="engine_google")],
            [InlineKeyboardButton("Bark", callback_data="engine_bark")],
            [InlineKeyboardButton("XTTS", callback_data="engine_xtts")]
        ]

        await query.message.reply_text(
            "Choose Voice Engine:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif data.startswith("engine_"):
        engine_name = data.replace("engine_", "")
        user_engine[uid] = engine_name

        await query.message.reply_text(
            f"Selected Engine: {engine_name}"
        )

    elif data == "my_files":
        await query.message.reply_text(
            "My Files feature coming next phase."
        )

    elif data == "settings":
        await query.message.reply_text(
            "Settings panel coming next phase."
        )


# =========================
# TEXT TO VOICE
# =========================
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.message.from_user.id
    text = update.message.text

    if user_mode.get(uid) != "await_text":
        return

    engine = user_engine.get(uid, "edge")

    await update.message.reply_text("Generating voice...")

    file_name = f"{uid}.mp3"

    # EDGE TTS
    if engine == "edge":
        communicate = edge_tts.Communicate(
            text=text,
            voice="en-US-AriaNeural"
        )
        await communicate.save(file_name)

    # GTTS
    else:
        tts = gTTS(text=text, lang="en")
        tts.save(file_name)

    await update.message.reply_audio(
        audio=open(file_name, "rb")
    )

    user_mode[uid] = None


# =========================
# REGISTER HANDLERS
# =========================
def setup_handlers(app):
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            text_handler
        )
    )