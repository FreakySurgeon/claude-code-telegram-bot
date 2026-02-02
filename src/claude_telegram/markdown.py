"""Convert Markdown to Telegram HTML."""

import re
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
