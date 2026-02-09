"""Tests for request queue."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_telegram.queue import QueueItem, RequestQueue


@pytest.fixture
def queue():
    return RequestQueue(maxsize=3)


def test_queue_item_creation():
    """Test QueueItem dataclass."""
    item = QueueItem(
        prompt="Hello",
        source="telegram",
        chat_id="123",
    )
    assert item.prompt == "Hello"
    assert item.retry_count == 0
    assert item.original_error is None


def test_queue_item_with_retry():
    """Test QueueItem retry creation."""
    original = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    retry = original.as_retry("Timed out after 300s")
    assert retry.retry_count == 1
    assert retry.original_error == "Timed out after 300s"
    assert "[RETRY]" in retry.prompt
    assert "Hello" in retry.prompt


def test_queue_item_max_retries():
    """Test QueueItem won't retry more than once."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123", retry_count=1)
    assert item.can_retry is False


def test_queue_item_can_retry():
    """Test QueueItem can retry on first attempt."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123", retry_count=0)
    assert item.can_retry is True


@pytest.mark.asyncio
async def test_enqueue_and_dequeue(queue):
    """Test basic enqueue/dequeue."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    added = await queue.enqueue(item)
    assert added is True
    assert queue.size == 1

    dequeued = await queue.dequeue()
    assert dequeued.prompt == "Hello"
    assert queue.size == 0


@pytest.mark.asyncio
async def test_enqueue_full_queue(queue):
    """Test enqueue on full queue returns False."""
    for i in range(3):
        await queue.enqueue(QueueItem(prompt=f"msg{i}", source="telegram", chat_id="123"))

    added = await queue.enqueue(QueueItem(prompt="overflow", source="telegram", chat_id="123"))
    assert added is False


@pytest.mark.asyncio
async def test_drain_clears_queue(queue):
    """Test drain removes all items."""
    for i in range(3):
        await queue.enqueue(QueueItem(prompt=f"msg{i}", source="telegram", chat_id="123"))

    count = queue.drain()
    assert count == 3
    assert queue.size == 0
