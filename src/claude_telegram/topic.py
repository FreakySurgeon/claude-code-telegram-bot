"""Topic naming logic — provisional names, title extraction, Ollama fallback."""

import logging
import re
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

MAX_TOPIC_NAME = 128
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_TIMEOUT = 10


def _today_prefix(*, is_agent: bool = False) -> str:
    """Return '[Agent] DD/MM - ' or 'DD/MM - ' prefix for today."""
    d = datetime.now().strftime("%d/%m")
    if is_agent:
        return f"[Agent] {d} - "
    return f"{d} - "


def _strip_command(message: str) -> str:
    """Strip leading /command from message text."""
    return re.sub(r"^/\S+\s*", "", message).strip()


def generate_provisional_name(message: str, *, is_agent: bool = False) -> str:
    """Generate a provisional topic name from the first message.

    Format: '[Agent] DD/MM - message...' or 'DD/MM - message...'
    Truncates so total length <= 128 chars (Telegram limit).
    """
    text = _strip_command(message)
    if not text:
        text = "Nouvelle conversation"

    prefix = _today_prefix(is_agent=is_agent)
    max_text_len = MAX_TOPIC_NAME - len(prefix)

    if len(text) > max_text_len:
        text = text[: max_text_len - 3] + "..."

    return prefix + text


def extract_title_from_response(response: str) -> tuple[str, str | None]:
    """Extract a <!-- title: ... --> comment from Claude's response.

    Returns (cleaned_text, title) or (original_response, None).
    """
    match = re.search(r"<!--\s*title:\s*(.+?)\s*-->", response)
    if not match:
        return (response, None)

    title = match.group(1).strip()
    cleaned = response[: match.start()] + response[match.end() :]
    cleaned = cleaned.strip()
    return (cleaned, title)


async def generate_title_fallback(message: str, response: str) -> str:
    """Generate a short title via Ollama (qwen3:4b) as fallback.

    On any error, returns a truncated version of the message (50 chars + '...').
    """
    truncated_msg = message[:200]
    truncated_resp = response[:300]

    prompt = (
        "Résume cette conversation en 5 mots maximum "
        "(en français, sans ponctuation) :\n\n"
        f"Utilisateur : {truncated_msg}\n"
        f"Assistant : {truncated_resp}"
    )

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 20},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        title = data.get("response", "").strip()
        # Clean up: strip quotes, periods, trailing punctuation
        title = title.strip("\"'«»")
        title = re.sub(r"[.!?,:;]+$", "", title).strip()

        if not title:
            raise ValueError("Empty title from Ollama")

        return title[:MAX_TOPIC_NAME]

    except Exception as exc:
        logger.warning("Ollama title generation failed: %s", exc)
        fallback = _strip_command(message).strip() or "Conversation"
        if len(fallback) > 50:
            fallback = fallback[:50] + "..."
        return fallback


def format_topic_name(title: str, *, is_agent: bool = False) -> str:
    """Format a final topic name with date prefix.

    Format: '[Agent] DD/MM - title' or 'DD/MM - title'
    Truncates to fit within 128 chars.
    """
    prefix = _today_prefix(is_agent=is_agent)
    max_title_len = MAX_TOPIC_NAME - len(prefix)

    if len(title) > max_title_len:
        title = title[: max_title_len - 3] + "..."

    return prefix + title
