"""Send an AI answer to Telegram: formatted, chunked, never rejected."""
from __future__ import annotations

from aiogram.types import Message

from app.utils.logging import get_logger
from app.utils.telegram_format import to_telegram_chunks

log = get_logger(__name__)


async def send_reply(message: Message, text: str) -> None:
    pairs = to_telegram_chunks(text or "(empty response)")
    for html_text, plain in pairs or [("(empty response)", "(empty response)")]:
        try:
            await message.answer(html_text, parse_mode="HTML")
        except Exception as e:  # Telegram rejected the markup -> send as plain text
            log.warning("html_send_failed err=%s; falling back to plain", e)
            await message.answer(plain, parse_mode=None)
