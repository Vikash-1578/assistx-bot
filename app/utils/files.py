"""File handling utilities — safe temp storage, validation, cleanup."""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".json", ".csv",
    ".pdf", ".docx", ".log", ".html", ".xml",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".json", ".csv", ".log", ".html", ".xml",
}


def is_allowed_extension(filename: str) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_text_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in TEXT_EXTENSIONS


def safe_temp_path(suffix: str) -> Path:
    """Create a unique temp file path with the given suffix."""
    fd, name = tempfile.mkstemp(suffix=suffix or ".bin", prefix="tgbot_")
    path = Path(name)
    # Close fd so callers can write to it
    import os
    os.close(fd)
    return path


def cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        log.warning("temp_cleanup_failed path=%s err=%s", path, e)


def read_text_safely(path: Path, max_chars: int = 8000) -> str:
    """Read a text file with size cap. Never raises."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n... [truncated]"
        return raw
    except Exception as e:
        log.warning("read_text_failed path=%s err=%s", path, e)
        return ""


# ------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

ALLOWED_IMAGE_MIMES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def validate_image_size(size: int | None) -> bool:
    """Return True if image is within allowed size."""
    if size is None:
        return True
    return size <= MAX_IMAGE_BYTES
