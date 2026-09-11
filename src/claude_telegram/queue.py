"""Request queue for serializing Claude requests."""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from .email_prompt import should_notify_email
from .ports import ConversationRef, Event
from .routing import event_type_for

logger = logging.getLogger(__name__)

RETRY_PREFIX = (
    "[RETRY] La requête précédente a été interrompue après 10 minutes "
    "(probablement un appel MCP bloqué). Adapte ta stratégie : "
    "évite les recherches longues, limite les appels MCP, va à l'essentiel.\n\n---\n\n"
)

DEAD_LETTER_MAX = 50


def _write_dead_letter(item: "QueueItem", error: str) -> None:
    """Persist failed items to dead-letter.json for later review."""
    try:
        working_dir = os.environ.get("GTD_WORKING_DIR", "")
        if not working_dir:
            logger.warning("GTD_WORKING_DIR not set, skipping dead-letter write")
            return
        dl_path = Path(working_dir) / "data" / "dead-letter.json"
        entries: list[dict] = []
        if dl_path.exists():
            try:
                entries = json.loads(dl_path.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": item.source,
            "type": item.metadata.get("reminder_type", item.source),
            "prompt_preview": item.prompt[:200],
            "error": str(error),
            "model": item.model or "default",
        })
        if len(entries) > DEAD_LETTER_MAX:
            entries = entries[-DEAD_LETTER_MAX:]
        dl_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    except Exception:
        logger.warning("Failed to write dead-letter entry", exc_info=True)


@dataclass
class QueueItem:
    """A request queued for Claude processing."""
    prompt: str
    source: Literal["telegram", "email", "cron", "zulip", "fitness", "calendar"] | str
    chat_id: str | None = None  # Telegram legacy; other channels use `conversation`
    # Optional metadata
    metadata: dict = field(default_factory=dict)
    model: str | None = None
    continue_session: bool = False
    bypass_permissions: bool = True
    new_session: bool = False
    allowed_tools: list[str] | None = None
    timeout: float = 600  # seconds (10 min default, override for email/cron)
    # Retry state
    retry_count: int = 0
    original_error: str | None = None
    thread_id: int | None = None
    # Channel-independent routing (set by producers; None = legacy Telegram fields)
    conversation: ConversationRef | None = None
    event_type: str | None = None
    channel_context: str | None = None

    def ref_or_legacy(self, bot_name: str | None = "gtd") -> ConversationRef | None:
        """Where replies go: the conversation, else the legacy Telegram chat/thread.

        Emails have no conversation until something is worth notifying.
        """
        if self.conversation is not None:
            return self.conversation
        if self.source == "email" or not self.chat_id:
            return None
        return ConversationRef("telegram", str(self.chat_id), thread_id=self.thread_id, bot=bot_name)

    @property
    def can_retry(self) -> bool:
        return self.retry_count < 1

    def as_retry(self, error: str) -> "QueueItem":
        """Create a retry copy with enriched prompt."""
        return QueueItem(
            prompt=f"{RETRY_PREFIX}{self.prompt}",
            source=self.source,
            chat_id=self.chat_id,
            metadata=self.metadata,
            model=self.model,
            continue_session=False,
            bypass_permissions=self.bypass_permissions,
            new_session=True,
            allowed_tools=self.allowed_tools,
            timeout=self.timeout,
            retry_count=self.retry_count + 1,
            original_error=error,
            thread_id=self.thread_id,
            conversation=self.conversation,
            event_type=self.event_type,
            channel_context=self.channel_context,
        )


class RequestQueue:
    """FIFO queue for Claude requests."""

    def __init__(self, maxsize: int = 10):
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=maxsize)

    async def enqueue(self, item: QueueItem) -> bool:
        """Add item to queue. Returns False if full."""
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            return False

    async def dequeue(self) -> QueueItem:
        """Get next item (blocks until available)."""
        return await self._queue.get()

    def drain(self) -> int:
        """Remove all items from queue. Returns count of removed items."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()


class PersistentQueue:
    """File-based persistent queue for messages during API unavailability.

    Each item is stored as a JSON file in the queue directory.
    Filenames are timestamped for FIFO ordering.
    Cron items are deduplicated by reminder_type.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def save(self, item: QueueItem) -> Path:
        """Persist a queue item to disk. Returns the file path."""
        reminder_type = item.metadata.get("reminder_type", "")

        # Dedup crons: remove existing file for same cron type
        # Skip dedup for calendar-action (each action is unique)
        if item.source == "cron" and reminder_type and reminder_type != "calendar-action":
            for existing in self.queue_dir.glob(f"*-cron-{reminder_type}.json"):
                existing.unlink()

        # Build filename (nanosecond precision to avoid collisions)
        ts = f"{time.time_ns()}"
        if item.source == "cron" and reminder_type:
            filename = f"{ts}-cron-{reminder_type}.json"
        else:
            filename = f"{ts}-{item.source}.json"

        data = {
            "prompt": item.prompt,
            "source": item.source,
            "chat_id": item.chat_id,
            "metadata": item.metadata,
            "model": item.model,
            "continue_session": item.continue_session,
            "bypass_permissions": item.bypass_permissions,
            "new_session": item.new_session,
            "allowed_tools": item.allowed_tools,
            "timeout": item.timeout,
            "thread_id": item.thread_id,
            "conversation": item.conversation.to_dict() if item.conversation else None,
            "event_type": item.event_type,
            "channel_context": item.channel_context,
            "queued_at": datetime.now().isoformat(),
        }

        path = self.queue_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return path

    def list_items(self) -> list[QueueItem]:
        """List all queued items in FIFO order."""
        return [item for item, _ in self.list_items_with_paths()]

    def list_items_with_paths(self) -> list[tuple[QueueItem, Path]]:
        """List all queued items with their file paths, in FIFO order.

        Corrupt files are skipped (logged as warning).
        """
        files = sorted(self.queue_dir.glob("*.json"))
        result = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                item = QueueItem(
                    prompt=data["prompt"],
                    source=data["source"],
                    chat_id=data["chat_id"],
                    metadata=data.get("metadata", {}),
                    model=data.get("model"),
                    continue_session=data.get("continue_session", False),
                    bypass_permissions=data.get("bypass_permissions", True),
                    new_session=data.get("new_session", True),
                    allowed_tools=data.get("allowed_tools"),
                    timeout=data.get("timeout", 600),
                    thread_id=data.get("thread_id"),
                    conversation=ConversationRef.from_dict(data["conversation"]) if data.get("conversation") else None,
                    event_type=data.get("event_type"),
                    channel_context=data.get("channel_context"),
                )
                result.append((item, f))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Skipping corrupt queue file {f}: {e}")
        return result

    def list_files(self) -> list[Path]:
        """List all queue files in FIFO order."""
        return sorted(self.queue_dir.glob("*.json"))

    def delete(self, path: Path):
        """Delete a processed queue file."""
        path.unlink(missing_ok=True)

    @property
    def size(self) -> int:
        return len(list(self.queue_dir.glob("*.json")))

    @property
    def is_empty(self) -> bool:
        return self.size == 0


class ApiStatus:
    """Track Claude API availability (in-memory only)."""

    def __init__(self):
        self.unavailable: bool = False
        self.since: datetime | None = None
        self.last_error: str | None = None

    def mark_unavailable(self, error: str):
        if not self.unavailable:
            logger.warning(f"Claude API marked unavailable: {error}")
        self.unavailable = True
        self.since = datetime.now()
        self.last_error = error

    def mark_available(self):
        if self.unavailable:
            logger.info("Claude API marked available again")
        self.unavailable = False
        self.since = None
        self.last_error = None


SCAN_TYPES = ("whatsapp", "gdrive-inbox", "sent-emails")


def _default_notifications(bot):
    """NotificationService used when none is wired (tests, early startup): Telegram only."""
    from . import state
    if state.notifications is not None:
        return state.notifications
    from .adapters.telegram.outbound import TelegramOutbound
    from .notifications import NotificationService
    from .routing import RoutingPolicy
    return NotificationService(RoutingPolicy.default(), {"telegram": TelegramOutbound({bot.name: bot})})


def _event_notifier(notifications):
    """LLM-provider alerts (fallback chain) published as ``llm_provider`` events."""
    async def _notify(text: str, severity: str) -> None:
        sev = severity if severity in ("urgent", "normal", "info") else "normal"
        prefix = "🚨 " if sev == "urgent" else ""
        await notifications.publish(Event("llm_provider", sev, body=prefix + text))
    return _notify


def _silent_event(item: QueueItem, text: str) -> Event | None:
    """Event for a silent source (email triage, periodic scans), or None to stay quiet.

    Scans keep the historical threshold (non-OK answer over 200 chars). Email
    triage only notifies urgent emails: proposals live in the inbox webapp.
    """
    reminder_type = item.metadata.get("reminder_type", "")
    text = text or ""
    if reminder_type in SCAN_TYPES:
        stripped = text.strip()
        if stripped and stripped.upper() != "OK" and len(stripped) > 200:
            return Event(item.event_type or event_type_for(reminder_type), "normal", body=text)
        return None
    if not should_notify_email(text):
        return None
    subject = item.metadata.get("subject", "(no subject)")
    return Event(item.event_type or "email_triage", "urgent", title=f"Email: {subject[:60]}", body=text)


async def _rename_default_topic(notifications, ref: ConversationRef, item: QueueItem, text: str,
                                session_store=None) -> ConversationRef:
    """Give a default topic ("(no topic)", "general chat"…) a real title after the first answer.

    Uses the ``<!-- title: … -->`` marker of the response, else a generated
    title. Only outbounds exposing ``default_topic_names`` (Zulip) qualify.
    """
    outbound = notifications.outbound(ref.channel) if hasattr(notifications, "outbound") else None
    defaults = getattr(outbound, "default_topic_names", None)
    if outbound is None or ref.topic is None or not defaults or ref.topic.lower() not in defaults:
        return ref
    from .topic import extract_title_from_response, generate_title_fallback
    _, title = extract_title_from_response(text)
    if not title:
        question = item.prompt.split("\n", 1)[-1] if item.prompt.startswith("[") else item.prompt
        title = await generate_title_fallback(question, text)
    try:
        new_ref = await outbound.rename_conversation(ref, title)
    except Exception:  # noqa: BLE001 — renaming is cosmetic
        logger.warning("Topic rename failed for %s", ref.key, exc_info=True)
        return ref
    if new_ref is None or new_ref == ref:
        return ref
    logger.info("Renamed %s -> %s", ref.key, new_ref.key)
    if session_store is not None:
        session_store.move(ref.key, new_ref.key)
    return new_ref


async def process_queue_item(
    item: QueueItem,
    runner,  # ClaudeRunner
    bot,     # BotConfig
    queue: "RequestQueue | None" = None,
    persistent_queue: "PersistentQueue | None" = None,
    api_status: "ApiStatus | None" = None,
    notifications=None,   # NotificationService
    session_store=None,   # conversations.SessionStore (non-Telegram sessions)
):
    """Process a single queue item: run Claude, handle timeout/retry, deliver the response.

    Replies go to the item's conversation (any channel); silent sources publish
    an Event only when there is something worth notifying.
    """
    notifications = notifications or _default_notifications(bot)
    session_name = runner.short_name
    reminder_type = item.metadata.get("reminder_type", "")
    silent = item.source == "email" or reminder_type in SCAN_TYPES
    ref = item.ref_or_legacy(getattr(bot, "name", None) or "gtd")

    progress = None
    if not silent and ref is not None:
        progress = await notifications.start_progress(
            ref,
            inbound_message_id=item.metadata.get("inbound_message_id"),
            continue_session=item.continue_session,
            session_name=session_name,
        )

    async def _stop_progress(ok: bool) -> None:
        nonlocal progress
        handle, progress = progress, None
        await notifications.stop_progress(handle, ok=ok)

    async def _say(text: str) -> None:
        """Short status line to the conversation (never for silent sources)."""
        if not silent and ref is not None:
            await notifications.reply(ref, text)

    system_prompt = getattr(bot, 'system_prompt', None)
    if item.channel_context:
        system_prompt = (system_prompt + "\n\n" if system_prompt else "") + item.channel_context

    _run_start = time.monotonic()
    try:
        logger.info(f"Processing queue item: source={item.source}, model={item.model or 'default'}, "
                     f"timeout={item.timeout}s, metadata={item.metadata}")
        from .providers import run_with_fallback
        result = await run_with_fallback(
            runner,
            item.prompt,
            notify=_event_notifier(notifications),
            model=item.model,
            continue_session=item.continue_session,
            new_session=item.new_session,
            allowed_tools=item.allowed_tools,
            bypass_permissions=item.bypass_permissions,
            system_prompt=system_prompt,
            mcp_config=getattr(bot, 'mcp_config_path', None),
            timeout=item.timeout,
        )

        # Check for quota error — persist and notify
        if result.is_quota_error and persistent_queue and api_status:
            was_available = not api_status.unavailable
            api_status.mark_unavailable(result.error or "unknown quota error")
            persistent_queue.save(item)
            await _stop_progress(ok=False)
            # First detection: prominent notification
            if was_available:
                await notifications.publish(Event(
                    "llm_provider", "urgent",
                    body="⚠️ **Crédits API Claude épuisés.**\n"
                         "Les messages sont automatiquement mis en file d'attente.\n"
                         "Traitement auto dès que les crédits seront restaurés.",
                ))
            # Per-message notification (only for non-silent sources)
            await _say(f"📥 Message en file d'attente (position {persistent_queue.size}).")
            return

        await _stop_progress(ok=not result.error)

        # If we got here with a successful result, clear unavailable flag
        if api_status and api_status.unavailable:
            api_status.mark_available()

        logger.info(f"Queue item completed: {item.source} (retry={item.retry_count}), response length={len(result.text)}")

        _run_duration = time.monotonic() - _run_start

        # --- Structured metrics logging ---
        from .metrics import write_metric
        from .config import settings
        _run_type = reminder_type or item.source
        write_metric(
            source=item.source,
            run_type=_run_type,
            model=item.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_creation_tokens=result.cache_creation_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            duration_s=_run_duration,
            duration_api_ms=result.duration_api_ms,
            status="ok",
            session_id=result.session_id,
            provider=result.provider,
            failure_kind=result.failure_kind,
        )

        # --- Token alert for crons ---
        if item.source == "cron" and settings.cron_token_alert_threshold:
            total_tokens = result.input_tokens + result.output_tokens
            if total_tokens > settings.cron_token_alert_threshold:
                alert_msg = (
                    f"⚠️ Cron **{_run_type}** a consommé "
                    f"**{total_tokens // 1000}k tokens** "
                    f"(seuil : {settings.cron_token_alert_threshold // 1000}k)"
                )
                if result.cost_usd:
                    alert_msg += f"\n\U0001f4b0 Coût : ${result.cost_usd:.2f}"
                await notifications.publish(Event("token_alert", "normal", body=alert_msg))

        # Update pending-actions status for calendar actions
        if reminder_type == "calendar-action":
            action_id = item.metadata.get("action_id")
            if action_id:
                from .pending_actions import update_status as update_action_status
                working_dir = getattr(bot, 'fixed_working_dir', None) or os.getcwd()
                pending_path = Path(working_dir) / "data" / "pending-actions.json"
                update_action_status(pending_path, action_id, "executed")
                logger.info(f"Calendar action {action_id} marked as executed")

        # Deliver the response (silent sources: selective output, no progress)
        if silent:
            event = _silent_event(item, result.text)
            if event is not None:
                await notifications.publish(event)
            elif reminder_type in SCAN_TYPES:
                logger.info(f"{reminder_type} scan silent (no notable action, len={len((result.text or '').strip())})")
                # Clean up session file to avoid polluting /resume history
                if result.session_id:
                    from .claude import delete_session
                    delete_session(result.session_id, runner.working_dir)
            else:
                logger.info(f"Email triage silent (no notification): {item.metadata.get('subject', '?')}")
        elif result.text:
            if ref is not None:
                await notifications.reply(ref, result.text, session_name=session_name)
            else:
                await notifications.publish(Event(item.event_type or "queue", "normal", body=result.text))
        elif item.source != "cron":
            await _say("_(pas de réponse)_")
        else:
            logger.info(f"Cron {reminder_type or '?'} produced no output, skipping notification")

        # Non-Telegram conversations keep their Claude session across restarts
        if ref is not None and ref.channel != "telegram" and not silent and result.text:
            ref = await _rename_default_topic(notifications, ref, item, result.text, session_store)
        if ref is not None and ref.channel != "telegram" and session_store is not None:
            session_store.save(ref.key, result.session_id or getattr(runner, "session_id", None))

        # --- Escalation detection ---
        # Agent can request a more powerful model via HTML markers
        response_text = result.text or ""
        if not item.metadata.get("escalated") and queue:
            escalate_to = None
            if "<!-- escalate:opus -->" in response_text:
                escalate_to = "opus"
            elif "<!-- escalate:sonnet -->" in response_text:
                escalate_to = "sonnet"
            elif "<!-- escalate:haiku -->" in response_text:
                escalate_to = "haiku"

            if escalate_to:
                logger.info(f"Escalation requested: {item.model or 'default'} → {escalate_to} (source={item.source})")
                # Build escalation context from previous agent's response
                summary = response_text.replace("<!-- escalate:sonnet -->", "").replace("<!-- escalate:opus -->", "").replace("<!-- escalate:haiku -->", "")
                summary = re.sub(r'<!--\s*buttons:\s*.+?\s*-->', '', summary).strip()
                if len(summary) > 2000:
                    summary = summary[:2000] + "\n[...tronqué]"

                escalated_item = QueueItem(
                    prompt=(
                        f"[ESCALADE depuis {item.model or 'sonnet'}]\n"
                        f"L'agent précédent a demandé l'escalade. Voici son résumé :\n\n"
                        f"{summary}\n\n"
                        f"---\n\n"
                        f"{item.prompt}"
                    ),
                    source=item.source,
                    chat_id=item.chat_id,
                    model=escalate_to,
                    metadata={**item.metadata, "escalated": True},
                    new_session=True,
                    bypass_permissions=item.bypass_permissions,
                    timeout=item.timeout,
                    thread_id=item.thread_id,
                    conversation=item.conversation,
                    event_type=item.event_type,
                    channel_context=item.channel_context,
                )
                await queue.enqueue(escalated_item)
                logger.info(f"Escalated item queued ({escalate_to})")

        # --- Post-session memory enrichment (loaded from external file) ---
        if item.source in ("telegram", "zulip") and result.text and len(result.text) > 100:
            from .main import _load_post_session_prompt
            post_prompt = _load_post_session_prompt()
            if post_prompt:
                try:
                    await runner.run(
                        post_prompt,
                        model="haiku",
                        continue_session=True,
                        bypass_permissions=True,
                        system_prompt=getattr(bot, 'system_prompt', None),
                        timeout=120,
                    )
                except Exception:
                    logger.warning("Session memory summary failed", exc_info=True)

        # --- GTD v2: Cron/email session continuity ---
        # Save session so user replies within 10min can resume the conversation
        if item.source in ("cron", "email") and result.text:
            runner.last_interaction = datetime.now()
            if result.session_id:
                runner.session_id = result.session_id

    except TimeoutError as e:
        logger.warning(f"Queue item timed out: {item.source} (retry={item.retry_count}, timeout={item.timeout}s)")
        _run_duration = time.monotonic() - _run_start
        from .metrics import write_metric
        write_metric(
            source=item.source,
            run_type=reminder_type or item.source,
            model=item.model,
            input_tokens=0, output_tokens=0, cost_usd=None,
            num_turns=0, duration_s=_run_duration, duration_api_ms=0,
            status="timeout", session_id=None,
        )
        await _stop_progress(ok=False)

        if item.can_retry and queue:
            retry_item = item.as_retry(str(e))
            await queue.enqueue(retry_item)
            timeout_min = int(item.timeout // 60)
            await _say(f"⏰ Timeout après {timeout_min}min — retry automatique en cours...")
        else:
            _write_dead_letter(item, f"TimeoutError after {item.retry_count + 1} attempts ({item.timeout}s)")
            await _say("❌ Échec après 2 tentatives (timeout). Requête abandonnée.")

    except Exception as e:
        _run_duration = time.monotonic() - _run_start
        from .metrics import write_metric
        write_metric(
            source=item.source,
            run_type=reminder_type or item.source,
            model=item.model,
            input_tokens=0, output_tokens=0, cost_usd=None,
            num_turns=0, duration_s=_run_duration, duration_api_ms=0,
            status="error", session_id=None,
        )
        await _stop_progress(ok=False)

        _write_dead_letter(item, str(e))
        logger.exception("Queue item processing error")
        try:
            await _say(f"❌ **Erreur:** `{e}`")
        except Exception:
            logger.warning("Could not report the error to the conversation", exc_info=True)

    finally:
        if progress is not None:
            await _stop_progress(ok=False)
