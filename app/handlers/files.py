"""File/document handler — reads uploaded documents and answers."""
from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

from app.services.agent_router import AgentRouter
from app.services.ai.models import TaskType
from app.services.memory import MemoryStore
from app.utils.files import (
    MAX_FILE_SIZE_BYTES,
    cleanup,
    is_allowed_extension,
    is_text_file,
    read_text_safely,
    safe_temp_path,
)
from app.utils.logging import get_logger

log = get_logger(__name__)
router = Router(name="files")

# Extensions that images handler owns — files handler must skip these
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _extract_text(path, filename: str) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if is_text_file(filename):
            return read_text_safely(path, max_chars=8000)
        if suffix == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            chunks: list[str] = []
            for page in reader.pages[:20]:
                chunks.append(page.extract_text() or "")
                if sum(len(c) for c in chunks) > 8000:
                    break
            return "\n".join(chunks)[:8000]
        if suffix == "docx":
            from docx import Document
            doc = Document(str(path))
            chunks = [p.text for p in doc.paragraphs[:200]]
            return "\n".join(chunks)[:8000]
    except Exception as e:
        log.warning("extract_failed filename=%s err=%s", filename, e)
    return ""


@router.message(F.document)
async def handle_document(
    message: Message,
    agent_router: AgentRouter,
    memory: MemoryStore,
) -> None:
    if not message.from_user or not message.document:
        return

    doc = message.document
    filename = doc.file_name or "file"
    size = doc.file_size or 0

    # Skip images — handled by images handler
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _IMAGE_EXTS:
        return

    if size > MAX_FILE_SIZE_BYTES:
        await message.answer(
            f"⚠ File too large ({size // 1024} KB). "
            f"Max: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )
        return

    if not is_allowed_extension(filename):
        await message.answer(
            "⚠ Unsupported file type.\n"
            "Allowed: txt, md, py, js, json, csv, pdf, docx"
        )
        return

    placeholder = await message.answer("📄 Reading file...")
    tmp_path = safe_temp_path(suffix="." + filename.rsplit(".", 1)[-1])

    try:
        if message.bot is None:
            await placeholder.edit_text("⚠ Bot unavailable.")
            return
        file = await message.bot.get_file(doc.file_id)
        if file.file_path is None:
            await placeholder.edit_text("⚠ Could not retrieve file.")
            return
        await message.bot.download_file(file.file_path, destination=tmp_path)

        text = _extract_text(tmp_path, filename)
        if not text.strip():
            await placeholder.edit_text("⚠ Could not extract text from this file.")
            return

        user_prompt = (message.caption or "").strip() or (
            "Analyze this file and give me a concise summary, plus key points."
        )
        full_prompt = f"{user_prompt}\n\n--- FILE: {filename} ---\n{text}"

        response = await agent_router.run(
            full_prompt,
            task=TaskType.SUMMARY,
            user_id=message.from_user.id,
            extra_system="You are analyzing an uploaded file for a Telegram user.",
        )

        reply = response.text or "(empty response)"
        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                await message.answer(reply[i : i + 4000])
        else:
            await message.answer(reply)

        try:
            await placeholder.delete()
        except Exception:
            pass
    except Exception as e:
        log.exception("file_handler_error user_id=%s err=%s", message.from_user.id, e)
        try:
            await placeholder.edit_text(
                "❌ Something went wrong while processing the file."
            )
        except Exception:
            pass
    finally:
        cleanup(tmp_path)
