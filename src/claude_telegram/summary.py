"""Short form of an event body, for mirrored routes.

A producer (a cron prompt, a pipeline) can embed a hand-written short version
between ``<!-- summary -->`` and ``<!-- /summary -->`` markers. Mirrored
channels receive that block; every other channel receives the body without it.
When no block is present, mirrored channels get a truncated copy of the body.
"""

from __future__ import annotations

import re

MIRROR_MAX_CHARS = 1200

_SUMMARY_RE = re.compile(r"<!--\s*summary\s*-->(.*?)<!--\s*/summary\s*-->", re.IGNORECASE | re.DOTALL)
_BLANK_RUNS_RE = re.compile(r"\n{3,}")


def split_summary(body: str) -> tuple[str, str | None]:
    """Return ``(body without the summary block, summary or None)``.

    A body without a (closed, non-empty) block is returned untouched.
    """
    match = _SUMMARY_RE.search(body or "")
    if not match:
        return body, None
    full = _BLANK_RUNS_RE.sub("\n\n", body[:match.start()] + body[match.end():]).strip()
    return full, match.group(1).strip() or None


def truncate_summary(text: str, limit: int = MIRROR_MAX_CHARS) -> str:
    """Cut ``text`` to at most ``limit`` chars, on a line boundary when one is close enough."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    newline = cut.rfind("\n")
    if newline > limit // 2:
        cut = cut[:newline]
    return cut.rstrip() + "…"
