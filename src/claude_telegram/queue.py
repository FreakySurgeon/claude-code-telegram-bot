"""Request queue for serializing Claude requests."""

import asyncio
import logging
from dataclasses import dataclass, field
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

    # Send animated status
    status = get_thinking_message()
    status_msg = await telegram.send_message(
        status,
        chat_id=item.chat_id,
        parse_mode="HTML",
        api_url=bot.api_url,
    )
    message_id = status_msg.get("result", {}).get("message_id")

    animation_task = None
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
        )

        # Stop animation + delete status
        if animation_task:
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass
        if message_id:
            await telegram.delete_message(item.chat_id, message_id, api_url=bot.api_url)

        # Send response
        if result.text:
            await send_response(result.text, item.chat_id, session_name=session_name, api_url=bot.api_url)
        else:
            await telegram.send_message("<i>(pas de réponse)</i>", chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url)

    except TimeoutError as e:
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
            await telegram.send_message(
                "⏰ Timeout après 5min — retry automatique en cours...",
                chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
            )
        else:
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
        await telegram.send_message(
            f"❌ <b>Erreur:</b> <code>{e}</code>",
            chat_id=item.chat_id, parse_mode="HTML", api_url=bot.api_url,
        )
