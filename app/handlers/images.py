"""Image handler — analyze photos AND image documents (uncompressed)."""
from __future__ import annotations

import base64
import io

from aiogram import F, Router
from aiogram.types import Message

from app.services.agent_router import AgentRouter
from app.services.ai.exceptions import AllProvidersFailed
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore
from app.utils.files import MAX_IMAGE_BYTES, validate_image_size
from app.utils.logging import get_logger

log = get_logger(__name__)
router = Router(name="images")

_DEFAULT_PROMPT = (
    "Analyze this image in detail. "
    "If there is text, transcribe it accurately. "
    "If it is code or terminal output, transcribe and explain it. "
    "If it is a chart/diagram, interpret the data. "
    "If it is a UI screenshot, describe and suggest improvements."
)

_IMAGE_MIMES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}

_IMAGE_EXT_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


async def _reply_chunks(message: Message, text: str) -> None:
    if len(text) <= 4000:
        await message.answer(text)
        return
    for i in range(0, len(text), 4000):
        await message.answer(text[i : i + 4000])


async def _analyze(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
    file_bytes: bytes,
    mime: str,
) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    caption = (message.caption or "").strip() or _DEFAULT_PROMPT
    b64 = base64.b64encode(file_bytes).decode("ascii")

    placeholder = await message.answer("🖼 Analyzing image...")

    try:
        response = await agent_router.run(
            caption,
            task=TaskType.VISION,
            user_id=user_id,
            images_b64=[b64],
            image_mime=mime,
        )
        memory.add_user(user_id, f"[Image] {caption}")
        memory.add_assistant(user_id, response.text)
        await _reply_chunks(message, response.text or "(empty response)")
        try:
            await placeholder.delete()
        except Exception:
            pass
    except AllProvidersFailed:
        log.error("vision_all_providers_failed user_id=%s", user_id)
        try:
            await placeholder.edit_text(
                "⚠ No vision-capable provider available. Try again."
            )
        except Exception:
            pass
    except Exception as e:
        log.exception("vision_handler_error user_id=%s err=%s", user_id, e)
        try:
            await placeholder.edit_text(
                "❌ Something went wrong while analyzing the image."
            )
        except Exception:
            pass


@router.message(F.photo)
async def handle_photo(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    if not message.from_user or not message.photo or message.bot is None:
        return
    photo = message.photo[-1]
    if not validate_image_size(photo.file_size):
        await message.answer(
            f"⚠ Image too large. Max: {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
        return
    try:
        buf = await message.bot.download(photo, destination=io.BytesIO())
        if buf is None:
            return
        await _analyze(message, agent_router, memory, buf.getvalue(), "image/jpeg")
    except Exception as e:
        log.exception("photo_download_failed err=%s", e)


@router.message(F.document)
async def handle_image_document(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    """Handle uncompressed images sent as documents (better quality)."""
    if not message.from_user or not message.document or message.bot is None:
        return

    doc = message.document
    filename = (doc.file_name or "").lower()
    mime = (doc.mime_type or "").lower()

    # Determine if this is an image
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1]
    detected_mime = _IMAGE_MIMES.get(mime) or _IMAGE_EXT_MIME.get(ext)
    if not detected_mime:
        # Not an image — let files handler deal with it
        return

    if not validate_image_size(doc.file_size):
        await message.answer(
            f"⚠ Image too large. Max: {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
        return

    try:
        buf = await message.bot.download(doc, destination=io.BytesIO())
        if buf is None:
            return
        await _analyze(message, agent_router, memory, buf.getvalue(), detected_mime)
    except Exception as e:
        log.exception("document_download_failed err=%s", e)
