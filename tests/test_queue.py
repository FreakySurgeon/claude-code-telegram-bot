"""Tests for request queue."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_telegram.queue import QueueItem, RequestQueue, process_queue_item, PersistentQueue, ApiStatus
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
    bot.use_queue = True
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


def test_queue_item_with_thread_id():
    """Test QueueItem includes thread_id."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123", thread_id=42)
    assert item.thread_id == 42


def test_queue_item_default_thread_id():
    """Test QueueItem defaults thread_id to None."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    assert item.thread_id is None


def test_queue_item_retry_preserves_thread_id():
    """Test that retry preserves thread_id."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123", thread_id=42)
    retry = item.as_retry("timeout")
    assert retry.thread_id == 42


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


@pytest.fixture
def pqueue(tmp_path):
    return PersistentQueue(tmp_path / "queue")


def test_persistent_queue_save_and_list(pqueue):
    """Test saving and listing queue items."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    pqueue.save(item)
    items = pqueue.list_items()
    assert len(items) == 1
    assert items[0].prompt == "Hello"


def test_persistent_queue_fifo_order(pqueue):
    """Test items are returned in FIFO order."""
    import time
    for i in range(3):
        pqueue.save(QueueItem(prompt=f"msg{i}", source="telegram", chat_id="123"))
        time.sleep(0.01)  # ensure different timestamps
    items = pqueue.list_items()
    assert [i.prompt for i in items] == ["msg0", "msg1", "msg2"]


def test_persistent_queue_delete(pqueue):
    """Test deleting a processed item."""
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    path = pqueue.save(item)
    assert pqueue.size > 0
    pqueue.delete(path)
    assert pqueue.size == 0


def test_persistent_queue_cron_dedup(pqueue):
    """Test cron deduplication replaces existing."""
    item1 = QueueItem(prompt="old scan", source="cron", chat_id="123", metadata={"reminder_type": "whatsapp"})
    item2 = QueueItem(prompt="new scan", source="cron", chat_id="123", metadata={"reminder_type": "whatsapp"})
    pqueue.save(item1)
    pqueue.save(item2)
    items = pqueue.list_items()
    assert len(items) == 1
    assert items[0].prompt == "new scan"


def test_persistent_queue_no_dedup_different_crons(pqueue):
    """Test different cron types are not deduped."""
    item1 = QueueItem(prompt="whatsapp", source="cron", chat_id="123", metadata={"reminder_type": "whatsapp"})
    item2 = QueueItem(prompt="morning", source="cron", chat_id="123", metadata={"reminder_type": "morning"})
    pqueue.save(item1)
    pqueue.save(item2)
    assert pqueue.size == 2


def test_persistent_queue_no_dedup_telegram(pqueue):
    """Test telegram messages are never deduped."""
    pqueue.save(QueueItem(prompt="msg1", source="telegram", chat_id="123"))
    pqueue.save(QueueItem(prompt="msg2", source="telegram", chat_id="123"))
    assert pqueue.size == 2


def test_persistent_queue_no_dedup_calendar_actions(pqueue):
    """Test calendar-action crons are NOT deduped (each action is unique)."""
    item1 = QueueItem(prompt="action1", source="cron", chat_id="123",
                      metadata={"reminder_type": "calendar-action", "action_id": "a1"})
    item2 = QueueItem(prompt="action2", source="cron", chat_id="123",
                      metadata={"reminder_type": "calendar-action", "action_id": "a2"})
    pqueue.save(item1)
    pqueue.save(item2)
    assert pqueue.size == 2


def test_persistent_queue_creates_directory(tmp_path):
    """Test queue creates directory if it doesn't exist."""
    pq = PersistentQueue(tmp_path / "nonexistent" / "queue")
    item = QueueItem(prompt="Hello", source="telegram", chat_id="123")
    pq.save(item)
    assert pq.size == 1


def test_api_status_default():
    """Test ApiStatus starts as available."""
    status = ApiStatus()
    assert status.unavailable is False
    assert status.since is None
    assert status.last_error is None


def test_api_status_mark_unavailable():
    """Test marking API as unavailable."""
    status = ApiStatus()
    status.mark_unavailable("quota exceeded")
    assert status.unavailable is True
    assert status.last_error == "quota exceeded"
    assert status.since is not None


def test_api_status_mark_available():
    """Test marking API as available again."""
    status = ApiStatus()
    status.mark_unavailable("quota exceeded")
    status.mark_available()
    assert status.unavailable is False
    assert status.since is None
    assert status.last_error is None


@pytest.mark.asyncio
async def test_process_queue_item_quota_error_persists(mock_bot, tmp_path):
    """Test that quota error persists item to disk and sets unavailable."""
    pqueue = PersistentQueue(tmp_path / "queue")
    api_status = ApiStatus()

    item = QueueItem(prompt="Hello", source="telegram", chat_id="12345")
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=ClaudeResult(
        text="", error="quota exceeded", is_quota_error=True,
    ))
    mock_runner.short_name = "gtd"

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}) as mock_send, \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨"):
        await process_queue_item(item, mock_runner, mock_bot,
                                 persistent_queue=pqueue, api_status=api_status)

    assert api_status.unavailable is True
    assert pqueue.size == 1
    # Should have sent the first-detection notification
    calls = [str(c) for c in mock_send.call_args_list]
    assert any("épuisés" in c for c in calls)


@pytest.mark.asyncio
async def test_process_queue_item_success_clears_unavailable(mock_bot, tmp_path):
    """Test that successful processing clears the unavailable flag."""
    pqueue = PersistentQueue(tmp_path / "queue")
    api_status = ApiStatus()
    api_status.mark_unavailable("quota exceeded")

    item = QueueItem(prompt="Hello", source="telegram", chat_id="12345")
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=ClaudeResult(text="Response", permission_denials=[]))
    mock_runner.short_name = "gtd"

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨"):
        await process_queue_item(item, mock_runner, mock_bot,
                                 persistent_queue=pqueue, api_status=api_status)

    assert api_status.unavailable is False


@pytest.mark.asyncio
async def test_full_unavailability_and_recovery_flow(mock_bot, tmp_path):
    """Test: quota error → persist → recovery → replay."""
    pqueue = PersistentQueue(tmp_path / "queue")
    status = ApiStatus()
    q = RequestQueue(maxsize=10)

    # Phase 1: API returns quota error
    item1 = QueueItem(prompt="Hello", source="telegram", chat_id="12345")
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=ClaudeResult(
        text="", error="quota exceeded", is_quota_error=True,
    ))
    mock_runner.short_name = "gtd"

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨"):
        await process_queue_item(item1, mock_runner, mock_bot, queue=q,
                                 persistent_queue=pqueue, api_status=status)

    assert status.unavailable is True
    assert pqueue.size == 1

    # Phase 2: API recovers — a new message succeeds
    mock_runner.run = AsyncMock(return_value=ClaudeResult(text="I'm back!", permission_denials=[]))
    item2 = QueueItem(prompt="New message", source="telegram", chat_id="12345")

    with patch("claude_telegram.telegram.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 2}}), \
         patch("claude_telegram.telegram.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.main.send_response", new_callable=AsyncMock) as mock_send, \
         patch("claude_telegram.main.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.main.get_thinking_message", return_value="✨"):
        await process_queue_item(item2, mock_runner, mock_bot, queue=q,
                                 persistent_queue=pqueue, api_status=status)

    # API should be marked as available again
    assert status.unavailable is False
    # Response was sent for the new message
    mock_send.assert_called()
    # Note: persistent queue items are replayed by queue_worker, not by process_queue_item.
    # The queue still has 1 item (the persisted one), but api_status is cleared.
