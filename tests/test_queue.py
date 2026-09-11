"""Tests for request queue."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_telegram.queue import QueueItem, RequestQueue, process_queue_item, PersistentQueue, ApiStatus
from claude_telegram.claude import ClaudeResult
from claude_telegram.ports import ConversationRef


class FakeNotifications:
    """Records what process_queue_item asks the NotificationService to do."""

    def __init__(self):
        self.replies = []      # (ref, text)
        self.published = []    # Event
        self.started = []      # (ref, kwargs)
        self.stopped = []      # (handle, ok)

    async def reply(self, ref, text, **kwargs):
        self.replies.append((ref, text))

    async def publish(self, event):
        self.published.append(event)

    async def start_progress(self, ref, **kwargs):
        self.started.append((ref, kwargs))
        return ("progress", ref.key)

    async def stop_progress(self, handle, ok=True):
        if handle is not None:
            self.stopped.append((handle, ok))


class FakeSessionStore:
    def __init__(self):
        self.saved = {}

    def save(self, key, session_id):
        self.saved[key] = session_id


def _runner(result):
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    runner.short_name = "gtd"
    runner.session_id = None
    runner.working_dir = "/tmp"
    return runner


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
    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}) as mock_tg_send, \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock) as mock_tg_del, \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock) as mock_send, \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
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

    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
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

    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨ <i>Thinking...</i>"):
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

    notifications = FakeNotifications()
    await process_queue_item(item, mock_runner, mock_bot, persistent_queue=pqueue,
                             api_status=api_status, notifications=notifications)

    assert api_status.unavailable is True
    assert pqueue.size == 1
    # First detection: an urgent event, plus the per-message queue notice in the conversation
    assert any(e.severity == "urgent" and "épuisés" in e.body for e in notifications.published)
    assert any("file d'attente" in text for _, text in notifications.replies)
    assert notifications.stopped == [(("progress", "telegram:12345:"), False)]


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

    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨"):
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

    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 1}}), \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨"):
        await process_queue_item(item1, mock_runner, mock_bot, queue=q,
                                 persistent_queue=pqueue, api_status=status)

    assert status.unavailable is True
    assert pqueue.size == 1

    # Phase 2: API recovers — the provider cooldown has expired, a new message succeeds
    from claude_telegram.providers import ProviderState, state_path
    ProviderState(state_path()).clear("claude")
    mock_runner.run = AsyncMock(return_value=ClaudeResult(text="I'm back!", permission_denials=[]))
    item2 = QueueItem(prompt="New message", source="telegram", chat_id="12345")

    with patch("claude_telegram.adapters.telegram.api.send_message", new_callable=AsyncMock, return_value={"result": {"message_id": 2}}), \
         patch("claude_telegram.adapters.telegram.api.delete_message", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.send_response", new_callable=AsyncMock) as mock_send, \
         patch("claude_telegram.adapters.telegram.outbound.animate_status", new_callable=AsyncMock), \
         patch("claude_telegram.adapters.telegram.outbound.get_thinking_message", return_value="✨"):
        await process_queue_item(item2, mock_runner, mock_bot, queue=q,
                                 persistent_queue=pqueue, api_status=status)

    # API should be marked as available again
    assert status.unavailable is False
    # Response was sent for the new message
    mock_send.assert_called()
    # Note: persistent queue items are replayed by queue_worker, not by process_queue_item.
    # The queue still has 1 item (the persisted one), but api_status is cleared.


@pytest.fixture
def clear_provider_state():
    from claude_telegram.providers import ProviderState, state_path
    ProviderState(state_path()).clear("claude")
    yield


@pytest.mark.asyncio
async def test_zulip_item_replies_on_its_conversation(mock_bot, clear_provider_state):
    ref = ConversationRef("zulip", "stream:quotidien", topic="Test canal")
    item = QueueItem(prompt="Salut", source="zulip", conversation=ref,
                     metadata={"inbound_message_id": "42"}, channel_context="Canal : Zulip")
    runner = _runner(ClaudeResult(text="Bonjour !", session_id="sess-1", permission_denials=[]))
    notifications, store = FakeNotifications(), FakeSessionStore()

    await process_queue_item(item, runner, mock_bot, notifications=notifications, session_store=store)

    assert notifications.replies == [(ref, "Bonjour !")]
    assert notifications.started[0][0] == ref
    assert notifications.started[0][1]["inbound_message_id"] == "42"
    assert notifications.stopped == [(("progress", ref.key), True)]
    assert store.saved == {ref.key: "sess-1"}
    assert "Canal : Zulip" in runner.run.call_args.kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_email_urgent_publishes_urgent_event(mock_bot, clear_provider_state):
    item = QueueItem(prompt="triage", source="email", metadata={"subject": "Impôts"})
    runner = _runner(ClaudeResult(text="Label : Claude/Urgent — payer avant ce soir", permission_denials=[]))
    notifications = FakeNotifications()

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    assert notifications.started == [] and notifications.replies == []
    [event] = notifications.published
    assert event.type == "email_triage" and event.severity == "urgent"
    assert event.title == "Email: Impôts"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "OK",
    "OK — email inconnu de la webapp",
    "[Claude/Action] Facture à classer",
    "Proposition enregistrée. " + "x" * 400,
])
async def test_email_non_urgent_publishes_nothing(mock_bot, clear_provider_state, text):
    item = QueueItem(prompt="triage", source="email", metadata={"subject": "Facture"})
    runner = _runner(ClaudeResult(text=text, permission_denials=[]))
    notifications = FakeNotifications()

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    assert notifications.published == [] and notifications.replies == [] and notifications.started == []


@pytest.mark.asyncio
async def test_whatsapp_scan_ok_publishes_nothing(mock_bot, clear_provider_state):
    item = QueueItem(prompt="scan", source="cron", metadata={"reminder_type": "whatsapp"})
    runner = _runner(ClaudeResult(text="OK", permission_denials=[]))
    notifications = FakeNotifications()

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    assert notifications.published == [] and notifications.replies == [] and notifications.started == []


@pytest.mark.asyncio
async def test_long_scan_result_publishes_typed_event(mock_bot, clear_provider_state):
    item = QueueItem(prompt="scan", source="cron", metadata={"reminder_type": "gdrive-inbox"})
    runner = _runner(ClaudeResult(text="x" * 300, permission_denials=[]))
    notifications = FakeNotifications()

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    [event] = notifications.published
    assert event.type == "gdrive_inbox" and event.severity == "normal"


@pytest.mark.asyncio
async def test_legacy_item_replies_on_telegram_ref(mock_bot, clear_provider_state):
    item = QueueItem(prompt="Salut", source="telegram", chat_id="12345", thread_id=7)
    runner = _runner(ClaudeResult(text="Réponse", permission_denials=[]))
    notifications = FakeNotifications()

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    [(ref, text)] = notifications.replies
    assert ref == ConversationRef("telegram", "12345", thread_id=7, bot="gtd")
    assert text == "Réponse"


def test_persistent_queue_roundtrip_with_conversation(pqueue):
    ref = ConversationRef("zulip", "dm:thomas@example.com")
    pqueue.save(QueueItem(prompt="Hi", source="zulip", conversation=ref,
                          event_type="conversation", channel_context="ctx"))
    [item] = pqueue.list_items()
    assert item.conversation == ref
    assert item.event_type == "conversation" and item.channel_context == "ctx"
    assert item.ref_or_legacy() == ref


class FakeTopicOutbound:
    default_topic_names = {"", "(no topic)", "general chat"}

    def __init__(self):
        self.renamed = []

    async def rename_conversation(self, ref, title):
        self.renamed.append((ref, title))
        return ref.with_topic(title)


@pytest.mark.asyncio
async def test_default_topic_renamed_from_title_marker(mock_bot, clear_provider_state):
    ref = ConversationRef("zulip", "stream:quotidien", topic="(no topic)")
    item = QueueItem(prompt="[zulip · #quotidien › (no topic) · de Thomas]\nOn fait les courses ?",
                     source="zulip", conversation=ref)
    runner = _runner(ClaudeResult(text="Oui.\n<!-- title: Courses samedi -->", session_id="s2",
                                  permission_denials=[]))
    notifications, store = FakeNotifications(), FakeSessionStore()
    store.move = MagicMock()
    out = FakeTopicOutbound()
    notifications.outbound = lambda channel: out

    await process_queue_item(item, runner, mock_bot, notifications=notifications, session_store=store)

    assert out.renamed == [(ref, "Courses samedi")]
    new_key = ref.with_topic("Courses samedi").key
    store.move.assert_called_once_with(ref.key, new_key)
    assert store.saved == {new_key: "s2"}


@pytest.mark.asyncio
async def test_named_topic_not_renamed(mock_bot, clear_provider_state):
    ref = ConversationRef("zulip", "stream:quotidien", topic="Courses")
    item = QueueItem(prompt="x", source="zulip", conversation=ref)
    runner = _runner(ClaudeResult(text="Oui.\n<!-- title: Autre -->", permission_denials=[]))
    notifications = FakeNotifications()
    out = FakeTopicOutbound()
    notifications.outbound = lambda channel: out

    await process_queue_item(item, runner, mock_bot, notifications=notifications)

    assert out.renamed == []
