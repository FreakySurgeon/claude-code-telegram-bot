"""Zulip InboundChannel: long-polls the bot's event queue.

Which messages reach the core:
- direct messages (when ``dm`` is enabled);
- every message of a ``listen_streams`` stream;
- messages of a ``mention_streams`` stream that @-mention the bot;
- any message in a topic the agent opened itself (``OwnedTopics``).

The queue id, last event id and the ids of messages already dispatched are
persisted in the state file after each batch, so a restart resumes the same
queue (or re-registers when Zulip has expired it) without double answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

import httpx

from ...ports import Attachment, ConversationRef, InboundHandler, InboundMessage
from .client import ZulipClient, ZulipError
from .outbound import OwnedTopics

logger = logging.getLogger(__name__)

SEEN_LIMIT = 500
BACKOFF_MAX = 60

_UPLOAD_RE = re.compile(r"\[([^\]]*)\]\((/user_uploads/[^)\s]+)\)")
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp"}
_AUDIO_EXT = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac", ".flac"}


def _kind_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXT:
        return "image"
    if suffix in _AUDIO_EXT:
        return "audio"
    return "file"


class ZulipInbound:
    name = "zulip"

    def __init__(
        self,
        client: ZulipClient,
        *,
        state_path: Path,
        listen_streams: list[str],
        mention_streams: list[str],
        dm: bool,
        owned: OwnedTopics,
        bot_email: str | None = None,
    ):
        self.client = client
        self.state_path = Path(state_path)
        self.listen_streams = {s.lower() for s in listen_streams or []}
        self.mention_streams = {s.lower() for s in mention_streams or []}
        self.dm = dm
        self.owned = owned
        self.bot_email = (bot_email or client.email or "").lower()
        self.bot_name: str | None = None
        self.bot_user_id: int | None = None
        self.queue_id: str | None = None
        self.last_event_id: int = -1
        self._seen: list[int] = []
        self._stopped = False
        self.load_state()

    # --- state -----------------------------------------------------------

    def _read(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def load_state(self) -> None:
        data = self._read()
        self.queue_id = data.get("queue_id") or None
        self.last_event_id = int(data.get("last_event_id", -1))
        self._seen = [int(i) for i in data.get("seen") or []][-SEEN_LIMIT:]

    def save_state(self) -> None:
        data = self._read()  # keeps the "owned" key written by OwnedTopics
        data.update(queue_id=self.queue_id, last_event_id=self.last_event_id, seen=self._seen)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".inbound.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.state_path)

    def seen(self, message_id: int | str) -> bool:
        try:
            return int(message_id) in self._seen
        except (TypeError, ValueError):
            return False

    def mark_seen(self, message_id: int | str) -> None:
        mid = int(message_id)
        if mid not in self._seen:
            self._seen.append(mid)
            self._seen = self._seen[-SEEN_LIMIT:]

    # --- filtering -------------------------------------------------------

    def _is_own(self, message: dict) -> bool:
        if self.bot_user_id is not None and message.get("sender_id") == self.bot_user_id:
            return True
        return bool(self.bot_email) and (message.get("sender_email") or "").lower() == self.bot_email

    def _strip_mention(self, text: str) -> str:
        if self.bot_name:
            text = re.sub(rf"@_?\*\*{re.escape(self.bot_name)}(\|\d+)?\*\*", "", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    def to_inbound(self, event: dict) -> InboundMessage | None:
        if event.get("type") != "message":
            return None
        message = event.get("message") or {}
        mid = message.get("id")
        if self._is_own(message):
            logger.info("Zulip: ignored own message %s", mid)
            return None
        if mid is None or self.seen(mid):
            return None

        flags = event.get("flags") or message.get("flags") or []
        mentioned = "mentioned" in flags
        if message.get("type") in ("private", "direct"):
            if not self.dm:
                return None
            recipients = message.get("display_recipient") or []
            emails = sorted(
                r["email"] for r in recipients
                if isinstance(r, dict) and r.get("email") and r["email"].lower() != self.bot_email
            )
            ref = ConversationRef("zulip", "dm:" + ",".join(emails))
        else:
            stream = message.get("display_recipient") or ""
            topic = message.get("subject") or ""
            ref = ConversationRef("zulip", f"stream:{stream}", topic=topic)
            wanted = (
                stream.lower() in self.listen_streams
                or (stream.lower() in self.mention_streams and mentioned)
                or ref.key in self.owned
            )
            if not wanted:
                return None

        content = message.get("content") or ""
        attachments = [
            Attachment(_kind_for(path), url=path, name=name or Path(path).name)
            for name, path in _UPLOAD_RE.findall(content)
        ]
        return InboundMessage(
            channel="zulip",
            conversation_id=ref.conversation_id,
            user=message.get("sender_email") or "",
            text=self._strip_mention(content),
            attachments=attachments,
            is_mention=mentioned,
            reply_to=ref,
            message_id=str(mid),
            user_name=message.get("sender_full_name") or "",
        )

    # --- attachments -----------------------------------------------------

    async def fetch_attachment(self, att: Attachment) -> Path:
        data = await self.client.download(att.url or "")
        suffix = Path(att.name or att.url or "").suffix
        fd, name = tempfile.mkstemp(prefix="zulip-", suffix=suffix)
        with open(fd, "wb") as fh:
            fh.write(data)
        return Path(name)

    # --- loop ------------------------------------------------------------

    def stop(self) -> None:
        self._stopped = True

    async def _identify(self) -> None:
        me = await self.client.get_me()
        self.bot_user_id = me.get("user_id")
        self.bot_name = me.get("full_name") or self.bot_name
        self.bot_email = (me.get("email") or self.bot_email).lower()

    async def run(self, handler: InboundHandler) -> None:
        self._stopped = False
        backoff = 1
        while not self._stopped:
            try:
                if self.bot_user_id is None:
                    await self._identify()
                if not self.queue_id:
                    self.queue_id, self.last_event_id = await self.client.register(("message",))
                    logger.info("Zulip: registered event queue %s", self.queue_id)
                    self.save_state()
                events = await self.client.get_events(self.queue_id, self.last_event_id)
                for event in events:
                    self.last_event_id = max(self.last_event_id, int(event.get("id", -1)))
                    msg = self.to_inbound(event)
                    if msg is None:
                        continue
                    self.mark_seen(msg.message_id)
                    try:
                        await handler(msg)
                    except Exception:  # noqa: BLE001 — one bad message must not stop the loop
                        logger.exception("Zulip: handler failed for message %s", msg.message_id)
                self.save_state()
                backoff = 1
            except asyncio.CancelledError:
                raise
            except ZulipError as exc:
                if exc.code == "BAD_EVENT_QUEUE_ID":
                    logger.info("Zulip: event queue %s expired, re-registering", self.queue_id)
                    self.queue_id = None
                    continue
                logger.warning("Zulip API error (%s), retry in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.warning("Zulip connection error (%s), retry in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
