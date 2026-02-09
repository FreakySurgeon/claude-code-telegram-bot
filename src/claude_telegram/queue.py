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
