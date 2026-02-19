"""Request queue for serializing Claude requests."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

logger = logging.getLogger(__name__)

RETRY_PREFIX = (
    "[RETRY] La requête précédente a été interrompue après 5 minutes "
    "(probablement un appel MCP bloqué). Adapte ta stratégie : "
    "évite les recherches longues, limite les appels MCP, va à l'essentiel.\n\n---\n\n"
)


@dataclass
class QueueItem:
    """A request queued for Claude processing."""
    prompt: str
    source: Literal["telegram", "email", "cron"]
    chat_id: str
    # Optional metadata
    metadata: dict = field(default_factory=dict)
    continue_session: bool = False
    bypass_permissions: bool = True
    new_session: bool = False
    allowed_tools: list[str] | None = None
    timeout: float = 300  # seconds (5 min default, override for email/cron)
    # Retry state
    retry_count: int = 0
    original_error: str | None = None

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
            continue_session=False,
            bypass_permissions=self.bypass_permissions,
            new_session=True,
            allowed_tools=self.allowed_tools,
            timeout=self.timeout,
            retry_count=self.retry_count + 1,
            original_error=error,
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


async def process_queue_item(
    item: QueueItem,
    runner,  # ClaudeRunner
    bot,     # BotConfig
    queue: "RequestQueue | None" = None,
):
    """Process a single queue item: run Claude, handle timeout/retry, send response."""
    # Lazy imports to avoid circular dependency
    from . import telegram
    from .main import send_response, animate_status, get_thinking_message

    session_name = runner.short_name
    silent = item.source == "email" or item.metadata.get("reminder_type") == "whatsapp"

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
        )
        message_id = status_msg.get("result", {}).get("message_id")

        if message_id:
            animation_task = asyncio.create_task(
                animate_status(item.chat_id, message_id, item.continue_session, session_name, api_url=bot.api_url)
            )

    try:
        result = await runner.run(
            item.prompt,
            continue_session=item.continue_session,
            new_session=item.new_session,
            allowed_tools=item.allowed_tools,
            bypass_permissions=item.bypass_permissions,
            system_prompt=getattr(bot, 'system_prompt', None),
            mcp_config=getattr(bot, 'mcp_config_path', None),
            timeout=item.timeout,
        )

        # Stop animation + delete status
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        logger.info(f"Queue item completed: {item.source} (retry={item.retry_count}), response length={len(result.text)}")

        # Send response (silent sources: no "thinking" animation, selective output)
        if silent:
            reminder_type = item.metadata.get("reminder_type", "")
            if reminder_type == "whatsapp":
                # WhatsApp scan: send result only if Claude produced meaningful output
                if result.text and len(result.text.strip()) > 20:
                    await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url)
                else:
                    logger.info(f"WhatsApp scan silent (no meaningful output)")
            elif result.text and "Claude/Urgent" in result.text:
                await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url)
            else:
                subject = item.metadata.get("subject", "?")
                logger.info(f"Email triage silent (no Telegram): {subject}")
        elif result.text:
            await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url)
        else:
            await telegram.send_message("<i>(pas de réponse)</i>", chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url)

        # --- Post-session memory enrichment (loaded from external file) ---
        if item.source == "telegram" and result.text and len(result.text) > 100:
            from .main import _load_post_session_prompt
            post_prompt = _load_post_session_prompt()
            if post_prompt:
                try:
                    await runner.run(
                        post_prompt,
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
                )
        else:
            if not silent:
                await telegram.send_message(
                    "❌ Échec après 2 tentatives (timeout). Requête abandonnée.",
                    chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
                )

    except Exception as e:
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        logger.exception("Queue item processing error")
        if not silent:
            await telegram.send_message(
                f"❌ <b>Erreur:</b> <code>{e}</code>",
                chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
            )
