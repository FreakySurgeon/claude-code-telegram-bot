"""Convert Markdown to Telegram HTML."""

import re
import json
import html
import logging

logger = logging.getLogger(__name__)


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-supported HTML.

    Telegram supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="">
    """
    # Log input for debugging
    if '<ide_opened_file' in text or '<system-reminder' in text:
        logger.warning(f"XML tags detected in input text (first 500 chars): {text[:500]}")

    # FIRST: Remove system tags WITH their content (these are internal Claude/IDE tags)
    # Pattern: <tagname ...>content</tagname> - remove entire block
    system_tags = ['ide_opened_file', 'system-reminder', 'antml:function_calls',
                   'antml:invoke', 'antml:parameter', 'tool_result', 'ide_selection']
    for tag in system_tags:
        pattern = rf'<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>'
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        # Also handle self-closing tags
        text = re.sub(rf'<{re.escape(tag)}[^>]*/>', '', text, flags=re.IGNORECASE)

    # THEN: Remove any remaining XML-like tags (orphan tags, unknown tags, etc.)
    # This catches anything we missed above
    text = re.sub(r'<[^>]+>', '', text)

    # Escape HTML entities first (but we'll unescape our tags later)
    text = html.escape(text)

    # Code blocks (``` ... ```) - must be done before inline code
    text = re.sub(
        r'```(\w*)\n(.*?)```',
        lambda m: f'<pre>{m.group(2)}</pre>',
        text,
        flags=re.DOTALL
    )

    # Inline code (` ... `)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic (*text* or _text_) - be careful not to match inside words
    text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<i>\1</i>', text)

    # Strikethrough (~~text~~)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # Headers (# text) - make them bold
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    return text


def safe_telegram_text(text: str) -> str:
    """
    Prepare text for Telegram, escaping special characters if not using parse_mode.
    """
    return html.escape(text)


_SYSTEM_TAGS = ['ide_opened_file', 'system-reminder', 'antml:function_calls',
                'antml:invoke', 'antml:parameter', 'tool_result', 'ide_selection']
_BUTTON_MARKER_RE = re.compile(r'<!--\s*buttons:\s*(.+?)\s*-->')


def strip_system_tags(text: str) -> str:
    """Remove internal Claude/IDE tags with their content."""
    for tag in _SYSTEM_TAGS:
        text = re.sub(rf'<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>', '', text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(rf'<{re.escape(tag)}[^>]*/>', '', text, flags=re.IGNORECASE)
    return text


def markdown_to_zulip(text: str) -> str:
    """Prepare Claude Markdown for Zulip (Markdown in, Markdown out, no HTML).

    Removes system tags and HTML comments (title/buttons/escalate/severity
    markers), turns headings into bold lines (chat style) and simple HTML
    formatting tags into their Markdown equivalent. Code blocks are kept.
    """
    text = strip_system_tags(text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith('```'):
            continue
        part = re.sub(r'^#{1,6}\s+(.+?)\s*#*\s*$', r'**\1**', part, flags=re.MULTILINE)
        part = re.sub(r'<br\s*/?>', '\n', part, flags=re.IGNORECASE)
        part = re.sub(r'</?(b|strong)>', '**', part, flags=re.IGNORECASE)
        part = re.sub(r'</?(i|em)>', '*', part, flags=re.IGNORECASE)
        part = re.sub(r'</?code>', '`', part, flags=re.IGNORECASE)
        parts[i] = part
    return re.sub(r'\n{3,}', '\n\n', ''.join(parts)).strip()


def extract_button_labels(text: str) -> tuple[str, list[str]]:
    """Extract a ``<!-- buttons: [...] -->`` marker as plain labels.

    Channel-neutral counterpart of the Telegram inline-keyboard builder:
    ``confirm`` → ["✅ Confirmer", "❌ Annuler"], ``none`` or no marker → [].
    """
    match = _BUTTON_MARKER_RE.search(text)
    if not match:
        return text, []
    raw = match.group(1).strip()
    cleaned = (text[:match.start()] + text[match.end():]).strip()
    if raw.lower() == "confirm":
        return cleaned, ["✅ Confirmer", "❌ Annuler"]
    try:
        labels = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return cleaned, []
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        return cleaned, labels[:8]
    return cleaned, []


def split_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks, trying to break at newlines."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(line) > chunk_size:
            # Line itself exceeds chunk_size — flush current, then hard-split the line
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), chunk_size):
                chunks.append(line[i:i + chunk_size])
        elif len(current) + len(line) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks
