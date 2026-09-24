"""Global error handler for unhandled exceptions."""
from __future__ import annotations

from aiogram import Router
from aiogram.types import ErrorEvent

from app.utils.logging import get_logger

log = get_logger(__name__)
router = Router(name="errors")


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    log.exception(
        "unhandled_exception update_id=%s err=%s",
        getattr(event.update, "update_id", None),
        event.exception,
    )
    # Try to notify user with a controlled message
    try:
        update = event.update
        msg = update.message or update.edited_message
        if msg is not None:
            await msg.answer(
                "❌ Something went wrong. Please try again."
            )
    except Exception:
        pass
    return True
