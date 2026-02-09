"""Tests for request queue."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_telegram.queue import QueueItem, RequestQueue, process_queue_item
from claude_telegram.claude import ClaudeResult


@pytest.fixture
def queue():
    return RequestQueue(maxsize=3)


@pytest.fixture
def mock_bot():
    """Create a mock GTD bot."""
    bot = MagicMock()
    bot.name = "gtd"
    bot.chat_id = "12345"
    bot.api_url = None
    bot.system_prompt = None
    bot.mcp_config_path = None
    bot.multi_session = False
    return bot


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


@pytest.mark.asyncio
async def test_process_queue_item_success(mock_bot):
    """Test processing a queue item successfully."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="12345")
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=ClaudeResult(text="Response", permission_denials=[]))
    mock_runner.short_name = "gtd"

    # Patch at the source so lazy imports pick up mocks
    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}) as mock_tg_send, \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock) as mock_tg_del, \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock) as mock_send, \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
        await process_queue_item(item, mock_runner, mock_bot)
        mock_runner.run.assert_called_once()
        mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_process_queue_item_timeout_retries(mock_bot):
    """Test that timeout triggers a retry."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="12345")
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(side_effect=TimeoutError("timed out"))
    mock_runner.short_name = "gtd"

    q = RequestQueue(maxsize=10)

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
        await process_queue_item(item, mock_runner, mock_bot, queue=q)

    assert q.size == 1
    retry_item = await q.dequeue()
    assert retry_item.retry_count == 1
    assert "[RETRY]" in retry_item.prompt


@pytest.mark.asyncio
async def test_process_queue_item_timeout_no_second_retry(mock_bot):
    """Test that a retry item doesn't retry again."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="12345", retry_count=1)
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(side_effect=TimeoutError("timed out"))
    mock_runner.short_name = "gtd"

    q = RequestQueue(maxsize=10)

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
        await process_queue_item(item, mock_runner, mock_bot, queue=q)

    assert q.size == 0
