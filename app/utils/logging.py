"""
Structured, rotating logging.

- stdout: human-readable
- logs/bot.log: rotating file handler
- Never logs secrets — `redact()` scrubs known key patterns.
"""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LoggingSettings

_REDACT_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(gsk_[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
    re.compile(r"(hf_[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9_\-\.]{10,})", re.IGNORECASE),
]

_REDACTED = "***REDACTED***"


def redact(text: str) -> str:
    if not text:
        return text
    for pat in _REDACT_PATTERNS:
        text = pat.sub(_REDACTED, text)
    return text


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            try:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
            except TypeError:
                pass
        return True


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(cfg: LoggingSettings) -> None:
    """Idempotent logging setup. Call once at startup."""
    root = logging.getLogger()
    if getattr(root, "_configured", False):
        return

    root.setLevel(cfg.level.upper())
    root.handlers.clear()

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    redactor = _RedactingFilter()

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(fmt)
    stdout.addFilter(redactor)
    root.addHandler(stdout)

    log_path = Path(cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fileh = RotatingFileHandler(
        log_path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    fileh.setFormatter(fmt)
    fileh.addFilter(redactor)
    root.addHandler(fileh)

    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    root._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
