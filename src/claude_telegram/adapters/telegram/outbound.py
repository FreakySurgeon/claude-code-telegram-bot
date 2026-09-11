"""Telegram outbound adapter: message rendering, status animation, TelegramOutbound.

Moved from main.py (step 1 of the channel ports & adapters migration); the
rendering helpers keep their historical signatures because the Telegram UI
handlers still call them directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Sequence

from ...bots import BotConfig
from ...markdown import markdown_to_telegram_html, split_text
from ...ports import Action, AckStatus, Attachment, Capabilities, ConversationRef
from ...topic import generate_provisional_name
from . import api as telegram

logger = logging.getLogger(__name__)

# Claude Code spinner words (from the CLI)
# Source: https://github.com/levindixon/tengu_spinner_words
SPINNER_VERBS = [
    "Accomplishing", "Actioning", "Actualizing", "Baking", "Booping", "Brewing",
    "Calculating", "Cerebrating", "Channelling", "Churning", "Clauding", "Coalescing",
    "Cogitating", "Combobulating", "Computing", "Concocting", "Conjuring", "Considering",
    "Contemplating", "Cooking", "Crafting", "Creating", "Crunching", "Deciphering",
    "Deliberating", "Determining", "Discombobulating", "Divining", "Doing", "Effecting",
    "Elucidating", "Enchanting", "Envisioning", "Finagling", "Flibbertigibbeting",
    "Forging", "Forming", "Frolicking", "Generating", "Germinating", "Hatching",
    "Herding", "Honking", "Hustling", "Ideating", "Imagining", "Incubating", "Inferring",
    "Jiving", "Manifesting", "Marinating", "Meandering", "Moseying", "Mulling",
    "Mustering", "Musing", "Noodling", "Percolating", "Perusing", "Philosophising",
    "Pondering", "Pontificating", "Processing", "Puttering", "Puzzling", "Reticulating",
    "Ruminating", "Scheming", "Schlepping", "Shimmying", "Shucking", "Simmering",
    "Smooshing", "Spelunking", "Spinning", "Stewing", "Sussing", "Synthesizing",
    "Thinking", "Tinkering", "Transmuting", "Unfurling", "Unravelling", "Vibing",
    "Wandering", "Whirring", "Wibbling", "Wizarding", "Working", "Wrangling",
]


def get_thinking_message() -> str:
    """Get a random thinking message with emoji."""
    verb = random.choice(SPINNER_VERBS)
    return f"✨ <i>{verb}...</i>"

def get_continue_message() -> str:
    """Get a random continue message with emoji."""
    verb = random.choice(SPINNER_VERBS)
    return f"🔄 <i>{verb}...</i>"


async def animate_status(chat_id: str, message_id: int, continue_session: bool, session_name: str, api_url: str | None = None, message_thread_id: int | None = None):
    """Animate the status message with rotating messages."""
    prefix = f"[<code>{session_name}</code>] " if session_name != "default" else ""
    try:
        while True:
            await asyncio.sleep(2.5)  # Update every 2.5 seconds
            status = get_continue_message() if continue_session else get_thinking_message()
            new_status = f"{prefix}{status}"
            try:
                await telegram.edit_message(message_id, new_status, chat_id, parse_mode="HTML", api_url=api_url)
            except Exception:
                pass  # Ignore edit errors (message may be deleted)
    except asyncio.CancelledError:
        pass


async def send_response(text: str, chat_id: str, chunk_size: int = 4000, session_name: str = "default", api_url: str | None = None, message_thread_id: int | None = None, skip_buttons: bool = False):
    """Send Claude's response with smart button detection."""
    if not text.strip():
        await telegram.send_message(
            "<i>(no output)</i>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=api_url,
            message_thread_id=message_thread_id,
        )
        return

    # Extract buttons from raw text (before markdown conversion)
    if skip_buttons:
        cleaned_text = text
        buttons = None
    else:
        cleaned_text, buttons, _button_type = extract_buttons_from_response(text)

    # Convert markdown to Telegram HTML
    html_text = markdown_to_telegram_html(cleaned_text)

    # Split into chunks if needed
    chunks = split_text(html_text, chunk_size)

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        reply_markup = buttons if (is_last and buttons) else None
        try:
            await telegram.send_message(
                chunk,
                chat_id=chat_id,
                parse_mode="HTML",
                reply_markup=reply_markup,
                api_url=api_url,
                message_thread_id=message_thread_id,
            )
        except Exception as e:
            # Fallback to plain text if HTML fails
            logger.warning(f"HTML parse failed, falling back to plain text: {e}")
            await telegram.send_message(
                text if len(chunks) == 1 else chunk,
                chat_id=chat_id,
                parse_mode=None,
                reply_markup=reply_markup,
                api_url=api_url,
                message_thread_id=message_thread_id,
            )
        if not is_last:
            await asyncio.sleep(0.5)


BUTTON_MARKER_RE = re.compile(r'<!--\s*buttons:\s*(.+?)\s*-->')


def _build_feedback_buttons() -> dict:
    """Build the default 👍/👎 feedback buttons."""
    return {"inline_keyboard": [[
        {"text": "👍", "callback_data": "feedback:up"},
        {"text": "👎", "callback_data": "feedback:down"},
    ]]}


def extract_buttons_from_response(text: str) -> tuple[str, dict | None, str | None]:
    """Extract <!-- buttons: ... --> marker from response text.

    Returns (cleaned_text, reply_markup, button_type).
    button_type: "custom", "confirm", "feedback", or None.
    """
    match = BUTTON_MARKER_RE.search(text)

    if not match:
        # No marker → default feedback buttons
        return (text, _build_feedback_buttons(), "feedback")

    raw = match.group(1).strip()
    cleaned = (text[:match.start()] + text[match.end():]).strip()

    if raw.lower() == "confirm":
        buttons = {"inline_keyboard": [[
            {"text": "✅ Confirmer", "callback_data": "reply:✅"},
            {"text": "❌ Annuler", "callback_data": "reply:❌"},
        ]]}
        return (cleaned, buttons, "confirm")

    if raw.lower() == "none":
        return (cleaned, None, None)

    # Try JSON array of labels
    try:
        labels = json.loads(raw)
        if isinstance(labels, list) and all(isinstance(l, str) for l in labels):
            rows: list[list[dict]] = []
            row: list[dict] = []
            for label in labels[:8]:
                cb_data = f"reply:{label}"
                # Telegram callback_data max 64 bytes
                if len(cb_data.encode("utf-8")) > 64:
                    cb_data = f"reply:{label[:20]}"
                row.append({"text": label, "callback_data": cb_data})
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            return (cleaned, {"inline_keyboard": rows}, "custom")
    except (json.JSONDecodeError, TypeError):
        pass

    # Couldn't parse → feedback fallback
    return (cleaned, _build_feedback_buttons(), "feedback")


def actions_to_keyboard(actions: Sequence[Action]) -> dict | None:
    """Inline keyboard with rows of 3 buttons (Telegram limits callback_data to 64 bytes)."""
    if not actions:
        return None
    rows: list[list[dict]] = []
    for action in actions:
        data = action.data
        if len(data.encode("utf-8")) > 64:
            data = data.encode("utf-8")[:64].decode("utf-8", "ignore")
        if not rows or len(rows[-1]) == 3:
            rows.append([])
        rows[-1].append({"text": action.label, "callback_data": data})
    return {"inline_keyboard": rows}


class TelegramOutbound:
    """OutboundChannel for the Telegram bots (GTD, dev).

    A conversation is a forum topic of the bot's chat; ``target["bot"]``
    picks the bot. Without a title the root of the chat is used.
    """

    name = "telegram"

    def __init__(self, bots: dict[str, BotConfig]):
        self.bots = bots

    def capabilities(self) -> Capabilities:
        return Capabilities(max_len=4000, supports_edit=True, supports_topics=True)

    def _bot(self, name: str | None, chat_id: str | None = None) -> BotConfig:
        if name and name in self.bots:
            return self.bots[name]
        if chat_id is not None:
            for bot in self.bots.values():
                if str(bot.chat_id) == str(chat_id):
                    return bot
        if "gtd" in self.bots:
            return self.bots["gtd"]
        return next(iter(self.bots.values()))

    def _bot_for_ref(self, ref: ConversationRef) -> BotConfig:
        return self._bot(ref.bot, ref.conversation_id)

    async def open_conversation(self, target: dict, title: str = "", *, owned: bool = False) -> ConversationRef:
        bot = self._bot(target.get("bot"))
        thread_id = None
        if title:
            topic_name = generate_provisional_name(title, is_agent=True)
            result = await telegram.create_forum_topic(bot.chat_id, topic_name, api_url=bot.api_url)
            thread_id = result["result"]["message_thread_id"]
        return ConversationRef("telegram", str(bot.chat_id), thread_id=thread_id, topic=title or None, bot=bot.name)

    async def send(
        self,
        ref: ConversationRef,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
        actions: Sequence[Action] = (),
        session_name: str | None = None,
    ) -> list[str]:
        bot = self._bot_for_ref(ref)
        ids: list[str] = []
        for att in attachments:
            if att.kind == "image" and att.path:
                res = await telegram.send_photo(att.path, chat_id=ref.conversation_id, api_url=bot.api_url,
                                                message_thread_id=ref.thread_id)
                mid = (res or {}).get("result", {}).get("message_id")
                if mid:
                    ids.append(str(mid))
        if actions:
            cleaned = extract_buttons_from_response(text)[0]
            html_text = markdown_to_telegram_html(cleaned)
            chunks = split_text(html_text, 4000)
            for i, chunk in enumerate(chunks):
                markup = actions_to_keyboard(actions) if i == len(chunks) - 1 else None
                res = await telegram.send_message(chunk, chat_id=ref.conversation_id, parse_mode="HTML",
                                                  reply_markup=markup, api_url=bot.api_url,
                                                  message_thread_id=ref.thread_id)
                mid = (res or {}).get("result", {}).get("message_id")
                if mid:
                    ids.append(str(mid))
            return ids
        await send_response(text, ref.conversation_id, session_name=session_name or "default",
                            api_url=bot.api_url, message_thread_id=ref.thread_id,
                            skip_buttons=session_name is None)
        return ids

    async def ack(self, ref: ConversationRef, message_id: str, status: AckStatus) -> None:
        return None  # Telegram progress is the animated status message

    async def edit(self, ref: ConversationRef, message_id: str, text: str) -> None:
        bot = self._bot_for_ref(ref)
        await telegram.edit_message(int(message_id), text, ref.conversation_id, parse_mode="HTML", api_url=bot.api_url)

    async def delete(self, ref: ConversationRef, message_id: str) -> None:
        bot = self._bot_for_ref(ref)
        await telegram.delete_message(ref.conversation_id, int(message_id), api_url=bot.api_url)

    async def start_progress(
        self,
        ref: ConversationRef,
        *,
        inbound_message_id: str | None = None,
        continue_session: bool = False,
        session_name: str | None = None,
    ) -> Any:
        bot = self._bot_for_ref(ref)
        status_msg = await telegram.send_message(
            get_thinking_message(), chat_id=ref.conversation_id, parse_mode="HTML",
            api_url=bot.api_url, message_thread_id=ref.thread_id,
        )
        message_id = (status_msg or {}).get("result", {}).get("message_id")
        if not message_id:
            return None
        task = asyncio.create_task(animate_status(
            ref.conversation_id, message_id, continue_session, session_name or "default",
            api_url=bot.api_url, message_thread_id=ref.thread_id,
        ))
        return (ref.conversation_id, message_id, task, bot.api_url)

    async def stop_progress(self, handle: Any, *, ok: bool = True) -> None:
        if not handle:
            return
        chat_id, message_id, task, api_url = handle
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await telegram.delete_message(chat_id, message_id, api_url=api_url)

    async def rename_conversation(self, ref: ConversationRef, title: str) -> ConversationRef:
        if ref.thread_id is None:
            return ref
        bot = self._bot_for_ref(ref)
        await telegram.edit_forum_topic(ref.conversation_id, ref.thread_id, title, api_url=bot.api_url)
        return ref.with_topic(title)
