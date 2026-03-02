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
    source: Literal["telegram", "email", "cron"]
    chat_id: str
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
            "continue_session": item.continue_session,
            "bypass_permissions": item.bypass_permissions,
            "new_session": item.new_session,
            "allowed_tools": item.allowed_tools,
            "timeout": item.timeout,
            "thread_id": item.thread_id,
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
                    continue_session=data.get("continue_session", False),
                    bypass_permissions=data.get("bypass_permissions", True),
                    new_session=data.get("new_session", True),
                    allowed_tools=data.get("allowed_tools"),
                    timeout=data.get("timeout", 600),
                    thread_id=data.get("thread_id"),
                )
                result.append((item, f))
            except (json.JSONDecodeError, KeyError) as e:
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


async def process_queue_item(
    item: QueueItem,
    runner,  # ClaudeRunner
    bot,     # BotConfig
    queue: "RequestQueue | None" = None,
    persistent_queue: "PersistentQueue | None" = None,
    api_status: "ApiStatus | None" = None,
):
    """Process a single queue item: run Claude, handle timeout/retry, send response."""
    # Lazy imports to avoid circular dependency
    from . import telegram
    from .main import send_response, animate_status, get_thinking_message

    session_name = runner.short_name
    silent = item.source == "email" or item.metadata.get("reminder_type") in ("whatsapp", "gdrive-inbox", "sent-emails")

    # Send animated status (skip for emails — no Telegram notification)
    message_id = None
    animation_task = None
    if not silent:
        status = get_thinking_message()
        status_msg = await telegram.send_message(
            status,
            chat_id=item.chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
            message_thread_id=item.thread_id,
        )
        message_id = status_msg.get("result", {}).get("message_id")

        if message_id:
            animation_task = asyncio.create_task(
                animate_status(item.chat_id, message_id, item.continue_session, session_name, api_url=bot.api_url, message_thread_id=item.thread_id)
            )

    try:
        logger.info(f"Processing queue item: source={item.source}, model={item.model or 'default'}, "
                     f"timeout={item.timeout}s, metadata={item.metadata}")
        _run_start = time.monotonic()
        result = await runner.run(
            item.prompt,
            model=item.model,
            continue_session=item.continue_session,
            new_session=item.new_session,
            allowed_tools=item.allowed_tools,
            bypass_permissions=item.bypass_permissions,
            system_prompt=getattr(bot, 'system_prompt', None),
            mcp_config=getattr(bot, 'mcp_config_path', None),
            timeout=item.timeout,
        )

        # Check for quota error — persist and notify
        if result.is_quota_error and persistent_queue and api_status:
            was_available = not api_status.unavailable
            api_status.mark_unavailable(result.error or "unknown quota error")
            persistent_queue.save(item)
            # Stop animation + delete status
            if animation_task:
                animation_task.cancel()
                try: await animation_task
                except asyncio.CancelledError: pass
            if message_id:
                await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)
            # First detection: prominent notification to main chat
            if was_available:
                await telegram.send_message(
                    "⚠️ <b>Crédits API Claude épuisés.</b>\n"
                    "Les messages sont automatiquement mis en file d'attente.\n"
                    "Traitement auto dès que les crédits seront restaurés.",
                    chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
                )
            # Per-message notification (only for non-silent sources)
            if not silent:
                await telegram.send_message(
                    f"📥 Message en file d'attente (position {persistent_queue.size}).",
                    chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
                    message_thread_id=item.thread_id,
                )
            return

        # Stop animation + delete status
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        # If we got here with a successful result, clear unavailable flag
        if api_status and api_status.unavailable:
            api_status.mark_available()

        logger.info(f"Queue item completed: {item.source} (retry={item.retry_count}), response length={len(result.text)}")

        _run_duration = time.monotonic() - _run_start

        # --- Structured metrics logging ---
        from .metrics import write_metric
        from .config import settings
        reminder_type = item.metadata.get("reminder_type", "")
        _run_type = reminder_type or item.source
        write_metric(
            source=item.source,
            run_type=_run_type,
            model=item.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            duration_s=_run_duration,
            duration_api_ms=result.duration_api_ms,
            status="ok",
            session_id=result.session_id,
        )

        # --- Token alert for crons ---
        if item.source == "cron" and settings.cron_token_alert_threshold:
            total_tokens = result.input_tokens + result.output_tokens
            if total_tokens > settings.cron_token_alert_threshold:
                alert_msg = (
                    f"\u26a0\ufe0f Cron <b>{_run_type}</b> a consomm\u00e9 "
                    f"<b>{total_tokens // 1000}k tokens</b> "
                    f"(seuil : {settings.cron_token_alert_threshold // 1000}k)"
                )
                if result.cost_usd:
                    alert_msg += f"\n\U0001f4b0 Co\u00fbt : ${result.cost_usd:.2f}"
                await telegram.send_message(
                    alert_msg,
                    chat_id=bot.chat_id,
                    parse_mode="HTML",
                    api_url=bot.api_url,
                )

        # Update pending-actions status for calendar actions
        if item.metadata.get("reminder_type") == "calendar-action":
            action_id = item.metadata.get("action_id")
            if action_id:
                from .pending_actions import update_status as update_action_status
                working_dir = getattr(bot, 'fixed_working_dir', None) or os.getcwd()
                pending_path = Path(working_dir) / "data" / "pending-actions.json"
                update_action_status(pending_path, action_id, "executed")
                logger.info(f"Calendar action {action_id} marked as executed")

        # Send response (silent sources: no "thinking" animation, selective output)
        if silent:
            reminder_type = item.metadata.get("reminder_type", "")
            if reminder_type in ("whatsapp", "gdrive-inbox", "sent-emails"):
                # Periodic scan: send result only if Claude took a notable action
                # Short responses like "OK", "Timestamp mis à jour" = nothing to report
                text = (result.text or "").strip()
                if text and text.upper() != "OK" and len(text) > 200:
                    await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url, message_thread_id=item.thread_id, skip_buttons=True)
                else:
                    logger.info(f"{reminder_type} scan silent (no notable action, len={len(text)})")
                    # Clean up session file to avoid polluting /resume history
                    if result.session_id:
                        from .claude import delete_session
                        delete_session(result.session_id, runner.working_dir)
            elif result.text and "Claude/Urgent" in result.text:
                await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url, message_thread_id=item.thread_id, skip_buttons=True)
            else:
                subject = item.metadata.get("subject", "?")
                logger.info(f"Email triage silent (no Telegram): {subject}")
        elif result.text:
            await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url, message_thread_id=item.thread_id)
        else:
            await telegram.send_message("<i>(pas de réponse)</i>", chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url, message_thread_id=item.thread_id)

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
                )
                await queue.enqueue(escalated_item)
                logger.info(f"Escalated item queued ({escalate_to})")

        # --- Post-session memory enrichment (loaded from external file) ---
        if item.source == "telegram" and result.text and len(result.text) > 100:
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
        reminder_type = item.metadata.get("reminder_type", "")
        write_metric(
            source=item.source,
            run_type=reminder_type or item.source,
            model=item.model,
            input_tokens=0, output_tokens=0, cost_usd=None,
            num_turns=0, duration_s=_run_duration, duration_api_ms=0,
            status="timeout", session_id=None,
        )
        # Stop animation
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        if item.can_retry and queue:
            retry_item = item.as_retry(str(e))
            await queue.enqueue(retry_item)
            if not silent:
                timeout_min = int(item.timeout // 60)
                await telegram.send_message(
                    f"⏰ Timeout après {timeout_min}min — retry automatique en cours...",
                    chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
                    message_thread_id=item.thread_id,
                )
        else:
            _write_dead_letter(item, f"TimeoutError after {item.retry_count + 1} attempts ({item.timeout}s)")
            if not silent:
                await telegram.send_message(
                    "❌ Échec après 2 tentatives (timeout). Requête abandonnée.",
                    chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
                    message_thread_id=item.thread_id,
                )

    except Exception as e:
        _run_duration = time.monotonic() - _run_start
        from .metrics import write_metric
        reminder_type = item.metadata.get("reminder_type", "")
        write_metric(
            source=item.source,
            run_type=reminder_type or item.source,
            model=item.model,
            input_tokens=0, output_tokens=0, cost_usd=None,
            num_turns=0, duration_s=_run_duration, duration_api_ms=0,
            status="error", session_id=None,
        )
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        _write_dead_letter(item, str(e))
        logger.exception("Queue item processing error")
        if not silent:
            await telegram.send_message(
                f"❌ <b>Erreur:</b> <code>{e}</code>",
                chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
                message_thread_id=item.thread_id,
            )
