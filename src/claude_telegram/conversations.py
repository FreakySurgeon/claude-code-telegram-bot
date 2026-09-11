"""ConversationService: turns an InboundMessage from any channel into a QueueItem.

One Claude session per conversation (``ConversationRef.key``). Sessions of
non-Telegram channels are persisted in a small JSON store so a restart keeps
the conversation going; after ``ttl_hours`` without activity a new session
starts. Telegram keeps its own UI (handlers) and does not go through here.
"""

from __future__ import annotations

import json
import logging
import shutil
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from .ports import Attachment, ConversationRef, InboundChannel, InboundMessage
from .queue import QueueItem
from .transcribe import transcribe_audio

logger = logging.getLogger(__name__)

QUEUE_FULL_TEXT = "⚠️ File pleine, réessaie dans quelques minutes."


def runner_key(ref: ConversationRef) -> int:
    """SessionManager thread id for a conversation.

    Telegram keeps its topic id; other channels get a stable negative id so
    they never collide with a Telegram topic.
    """
    if ref.channel == "telegram":
        return ref.thread_id or 0
    return -(zlib.crc32(ref.key.encode()) or 1)


class SessionStore:
    """``{conversation key: {"session_id", "last_interaction"}}`` persisted as JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)

    def get(self, key: str) -> dict | None:
        return self._read().get(key)

    def save(self, key: str, session_id: str | None) -> None:
        data = self._read()
        data[key] = {"session_id": session_id, "last_interaction": datetime.now().isoformat()}
        self._write(data)

    def move(self, old_key: str, new_key: str) -> None:
        data = self._read()
        if old_key in data:
            data[new_key] = data.pop(old_key)
            self._write(data)


def _place(ref: ConversationRef) -> str:
    if ref.conversation_id.startswith("stream:"):
        stream = ref.conversation_id[len("stream:"):]
        return f"#{stream} › {ref.topic}" if ref.topic else f"#{stream}"
    return "DM"


class ConversationService:
    def __init__(
        self,
        *,
        notifications,
        queue_submit: Callable[[QueueItem], Awaitable[int | None]],
        sessions_manager,
        working_dir: str,
        session_store: SessionStore,
        data_dir: Path,
        ttl_hours: float = 12,
        channel_contexts: dict[str, str] | None = None,
        inbound_channels: dict[str, InboundChannel] | None = None,
        transcribe=transcribe_audio,
    ):
        self.notifications = notifications
        self.queue_submit = queue_submit
        self.sessions = sessions_manager
        self.working_dir = working_dir
        self.store = session_store
        self.data_dir = Path(data_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.channel_contexts = channel_contexts or {}
        self.inbound_channels = inbound_channels or {}
        self.transcribe = transcribe

    # --- sessions --------------------------------------------------------

    def runner_for(self, ref: ConversationRef):
        runner = self.sessions.get_session(self.working_dir, thread_id=runner_key(ref))
        if not runner.session_id:
            entry = self.store.get(ref.key) or {}
            runner.session_id = entry.get("session_id")
        return runner

    def is_fresh(self, ref: ConversationRef) -> bool:
        entry = self.store.get(ref.key)
        if not entry or not entry.get("session_id"):
            return True
        try:
            last = datetime.fromisoformat(entry.get("last_interaction") or "")
        except ValueError:
            return True
        return datetime.now() - last > self.ttl

    # --- commands --------------------------------------------------------

    async def _command(self, msg: InboundMessage) -> bool:
        words = msg.text.strip().split()
        command = words[0].lower() if words else ""
        ref = msg.ref
        if command == "/new":
            self.store.save(ref.key, None)
            self.runner_for(ref).session_id = None
            await self.notifications.reply(ref, "🆕 Nouvelle session.")
            return True
        if command == "/status":
            runner = self.runner_for(ref)
            state = "en cours" if runner.is_running else "au repos"
            session = (runner.session_id or "aucune")[:8]
            await self.notifications.reply(ref, f"Session `{session}` — {state}.")
            return True
        if command == "/cancel":
            cancelled = await self.runner_for(ref).cancel()
            await self.notifications.reply(ref, "🛑 Annulé." if cancelled else "Rien à annuler.")
            return True
        return False

    # --- attachments -----------------------------------------------------

    async def _attachment_line(self, channel: str, att: Attachment) -> str:
        inbound = self.inbound_channels.get(channel)
        path = Path(att.path) if att.path else None
        if path is None:
            if inbound is None:
                return f"[Pièce jointe non récupérée : {att.name or att.url}]"
            path = await inbound.fetch_attachment(att)
        if att.kind == "image":
            return f"[Image jointe : {path}]"
        if att.kind == "audio":
            try:
                result = await self.transcribe(str(path))
                return f"[Vocal transcrit] {result.text}"
            except Exception as exc:  # noqa: BLE001 — keep the message, report the failure
                logger.warning("Transcription failed for %s: %s", path, exc)
                return f"[Vocal reçu, transcription impossible : {path}]"
        inbox = self.data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / (att.name or path.name)
        shutil.copy(path, dest)
        return f"[Fichier reçu : {dest}]"

    # --- entry point -----------------------------------------------------

    async def handle(self, msg: InboundMessage) -> None:
        ref = msg.ref
        if await self._command(msg):
            return

        lines = [msg.text] if msg.text else []
        for att in msg.attachments:
            try:
                lines.append(await self._attachment_line(msg.channel, att))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Attachment %s failed: %s", att.name or att.url, exc)
                lines.append(f"[Pièce jointe non récupérée : {att.name or att.url}]")
        if not lines:
            return

        author = msg.user_name or msg.user
        prompt = f"[{msg.channel} · {_place(ref)} · de {author}]\n" + "\n".join(lines)

        new = self.is_fresh(ref)
        runner = self.runner_for(ref)
        # Never let the runner fall back to --continue/latest session on disk:
        # a new conversation starts clean, an ongoing one resumes its own session.
        runner.session_id = None if new else (self.store.get(ref.key) or {}).get("session_id")
        item = QueueItem(
            prompt=prompt,
            source=msg.channel,
            chat_id=None,
            conversation=ref,
            continue_session=not new,
            new_session=new,
            model="sonnet",
            metadata={"inbound_message_id": msg.message_id, "type": f"{msg.channel}-conversation"},
            channel_context=self.channel_contexts.get(msg.channel),
        )
        position = await self.queue_submit(item)
        if position is None:
            await self.notifications.reply(ref, QUEUE_FULL_TEXT)
            outbound = self.notifications.outbound(ref.channel)
            if msg.message_id and outbound is not None:
                try:
                    await outbound.ack(ref, msg.message_id, "error")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ack failed on %s: %s", ref.channel, exc)
            return
        logger.info("%s message %s queued (position %s, new_session=%s)", msg.channel, msg.message_id, position, new)
