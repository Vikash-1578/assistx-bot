"""
Turn an LLM's Markdown into Telegram-safe HTML, and split long answers.

The bot runs with parse_mode=HTML, so raw model output containing `<`, `&`
or half-finished Markdown makes Telegram reject the message. Everything here
escapes first, then adds only the few tags Telegram supports.
"""
from __future__ import annotations

import html
import re

MAX_CHUNK = 3000  # markdown chars per message (HTML adds tags; limit is 4096)

_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.M)
_BULLET = re.compile(r"^([ \t]*)[-*+][ \t]+", re.M)
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_HR = re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$", re.M)


def md_to_html(text: str) -> str:
    """Markdown subset -> Telegram HTML (b, code, pre, a). Always valid."""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    def fence(m: re.Match) -> str:
        lang, body = m.group(1), m.group(2).rstrip("\n")
        cls = f' class="language-{html.escape(lang, quote=True)}"' if lang else ""
        return keep(f"<pre><code{cls}>{html.escape(body, quote=False)}</code></pre>")

    text = _FENCE.sub(fence, text.replace("\x00", ""))
    text = _INLINE_CODE.sub(
        lambda m: keep(f"<code>{html.escape(m.group(1), quote=False)}</code>"), text
    )
    text = _LINK.sub(
        lambda m: keep(
            f'<a href="{html.escape(m.group(2), quote=True)}">'
            f"{html.escape(m.group(1), quote=False)}</a>"
        ),
        text,
    )
    text = _HR.sub("──────────", text)
    text = html.escape(text, quote=False)
    text = _HEADING.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BOLD.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BULLET.sub(lambda m: f"{m.group(1)}• ", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def split_markdown(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split on line boundaries; keep ``` blocks balanced across chunks."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    fence_lang: str | None = None  # set while inside a code block

    def flush() -> None:
        nonlocal cur, size
        body = "\n".join(cur).strip("\n")
        if fence_lang is not None:
            body += "\n```"
        if body.strip():
            chunks.append(body)
        cur, size = [], 0
        if fence_lang is not None:
            cur.append(f"```{fence_lang}")
            size = len(cur[0]) + 1

    for line in text.split("\n"):
        # break very long lines
        pieces = [line[i:i + limit - 10] for i in range(0, len(line), limit - 10)] or [""]
        for piece in pieces:
            if size + len(piece) + 1 > limit and cur:
                flush()
            cur.append(piece)
            size += len(piece) + 1
            stripped = piece.strip()
            if stripped.startswith("```"):
                fence_lang = None if fence_lang is not None else stripped[3:].strip()
    body = "\n".join(cur).strip("\n")
    if body.strip():
        chunks.append(body)
    return chunks


def to_telegram_chunks(text: str, limit: int = MAX_CHUNK) -> list[tuple[str, str]]:
    """[(html, plain_fallback), ...] ready to send one after another."""
    return [(md_to_html(c), c) for c in split_markdown(text, limit)]
