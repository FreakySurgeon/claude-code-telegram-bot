"""Tests for FastAPI main application."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Must patch before importing app
with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "test_token",
    "TELEGRAM_CHAT_ID": "12345",
}):
    from claude_telegram.main import app
    from claude_telegram.adapters.telegram.handlers import (
        handle_message,
        handle_command,
        run_claude,
    )
    from claude_telegram.adapters.telegram.outbound import send_response
    from claude_telegram.bots import BotConfig
    import claude_telegram.state as state


client = TestClient(app)


def _make_dev_bot() -> BotConfig:
    """Create a dev BotConfig for testing."""
    return BotConfig(
        name="dev",
        token="test_token",
        chat_id="12345",
        use_queue=False,
        commands_whitelist=[
            "/start", "/help", "/c", "/continue", "/new", "/dir", "/dirs",
            "/repos", "/rmdir", "/compact", "/cancel", "/status",
        ],
    )


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "claude_running" in data


def test_webhook_empty_update():
    """Test webhook with empty update."""
    # Webhook needs bots dict populated with a dev bot
    bot = _make_dev_bot()
    with patch.object(state, "bots", {"dev": bot}):
        response = client.post("/webhook", json={})
        assert response.status_code == 200
        assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_handle_message_authorized(authorized_message):
    """Test handling authorized message in a topic."""
    bot = _make_dev_bot()
    # Add is_topic_message to skip topic creation
    msg = authorized_message["message"]
    msg["is_topic_message"] = True
    msg["message_thread_id"] = 42
    with patch("claude_telegram.adapters.telegram.handlers.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_message(msg, bot)
        mock_run.assert_called_once_with("Hello Claude", "12345", bot, continue_session=False, thread_id=42, new_session=False)


@pytest.mark.asyncio
async def test_handle_message_unauthorized(unauthorized_message):
    """Test handling unauthorized message."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_message(unauthorized_message["message"], bot)
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_empty_text():
    """Test handling message with no text."""
    bot = _make_dev_bot()
    message = {
        "chat": {"id": 12345},
        "text": "",
        "is_topic_message": True,
        "message_thread_id": 42,
    }
    with patch("claude_telegram.adapters.telegram.handlers.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_message(message, bot)
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_handle_command_start():
    """Test /start command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/start", "12345", bot)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "Commands" in call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_continue():
    """Test /c command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_command("/c fix the bug", "12345", bot)
        mock_run.assert_called_once_with("fix the bug", "12345", bot, continue_session=True, thread_id=None)


@pytest.mark.asyncio
async def test_handle_command_continue_alias():
    """Test /continue command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_command("/continue do something", "12345", bot)
        mock_run.assert_called_once_with("do something", "12345", bot, continue_session=True, thread_id=None)


@pytest.mark.asyncio
async def test_handle_command_continue_no_args():
    """Test /c command without arguments."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/c", "12345", bot)
        mock_send.assert_called_once()
        assert "Usage:" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_compact():
    """Test /compact command."""
    from claude_telegram.claude import ClaudeResult
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = False
    mock_runner.compact = AsyncMock(return_value=ClaudeResult(text="Compacted", permission_denials=[]))
    mock_runner.short_name = "test"
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock):
            with patch("claude_telegram.adapters.telegram.handlers.send_response", new_callable=AsyncMock) as mock_chunked:
                await handle_command("/compact", "12345", bot)
                mock_runner.compact.assert_called_once()
                mock_chunked.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_compact_while_busy():
    """Test /compact when Claude is running."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = True
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/compact", "12345", bot)
            assert "busy" in mock_send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_handle_command_cancel():
    """Test /cancel command."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.cancel = AsyncMock(return_value=True)
    mock_runner.short_name = "test"
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/cancel", "12345", bot)
            mock_runner.cancel.assert_called_once()
            assert "Cancelled" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_cancel_nothing():
    """Test /cancel when nothing is running."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.cancel = AsyncMock(return_value=False)
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/cancel", "12345", bot)
            assert "Nothing" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_status():
    """Test /status command."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = True
    mock_runner.is_in_conversation = MagicMock(return_value=True)
    mock_runner.short_name = "test"
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/status", "12345", bot)
            assert "Running" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_unknown():
    """Test unknown command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/invalid", "12345", bot)
        # Now hits the whitelist check, not the else branch
        assert "commande inconnue" in mock_send.call_args[0][0].lower() or "Unknown" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_dir_with_path():
    """Test /dir command with path."""
    bot = _make_dev_bot()
    mock_session = MagicMock()
    mock_session.is_running = False
    mock_session.is_in_conversation = MagicMock(return_value=False)
    mock_session.short_name = "myproject"
    with patch("claude_telegram.adapters.telegram.handlers.sessions") as mock_sessions:
        mock_sessions.switch_session = MagicMock(return_value=mock_session)
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dir /path/to/myproject", "12345", bot)
            mock_sessions.switch_session.assert_called_once_with("/path/to/myproject")
            assert "Switched" in mock_send.call_args[0][0]
            assert "myproject" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_dir_no_args():
    """Test /dir command without arguments shows directory browser."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/dir", "12345", bot)
        msg = mock_send.call_args[0][0]
        assert "Current" in msg


@pytest.mark.asyncio
async def test_handle_command_dirs():
    """Test /dirs command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.sessions") as mock_sessions:
        mock_sessions.list_dirs = MagicMock(return_value=[
            ("/path/to/project1", 2),
            ("/path/to/project2", 1),
        ])
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dirs", "12345", bot)
            message = mock_send.call_args[0][0]
            assert "Active Directories" in message
            assert "project1" in message
            assert "project2" in message


@pytest.mark.asyncio
async def test_handle_command_dirs_empty():
    """Test /dirs command with no sessions."""
    bot = _make_dev_bot()
    with patch("claude_telegram.adapters.telegram.handlers.sessions") as mock_sessions:
        mock_sessions.list_dirs = MagicMock(return_value=[])
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dirs", "12345", bot)
            assert "No active sessions" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_callback_dir_switch():
    """Test callback for directory switching."""
    from claude_telegram.adapters.telegram.handlers import handle_callback
    bot = _make_dev_bot()
    mock_session = MagicMock()
    mock_session.is_running = False
    mock_session.is_in_conversation = MagicMock(return_value=False)
    mock_session.short_name = "myproject"
    callback = {
        "id": "123",
        "data": "dir:/path/to/myproject",
        "message": {"chat": {"id": 12345}, "message_id": 999},
    }
    with patch("claude_telegram.adapters.telegram.handlers.sessions") as mock_sessions:
        mock_sessions.switch_session = MagicMock(return_value=mock_session)
        with patch("claude_telegram.adapters.telegram.handlers.telegram.answer_callback", new_callable=AsyncMock):
            with patch("claude_telegram.adapters.telegram.handlers.telegram.edit_message", new_callable=AsyncMock) as mock_edit:
                await handle_callback(callback, bot)
                mock_sessions.switch_session.assert_called_once_with("/path/to/myproject")
                assert "Switched" in mock_edit.call_args[0][1]


@pytest.mark.asyncio
async def test_run_claude_when_busy():
    """Test run_claude when already running."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = True
    mock_runner.short_name = "test"
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await run_claude("Hello", "12345", bot, continue_session=False)
            assert "busy" in mock_send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_run_claude_success():
    """Test successful Claude run."""
    from claude_telegram.claude import ClaudeResult
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = False
    mock_runner.run = AsyncMock(return_value=ClaudeResult(text="Claude response", permission_denials=[]))
    mock_runner.short_name = "test"
    mock_runner.context_shown = True  # Skip context check
    mock_runner.is_in_conversation = MagicMock(return_value=True)
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"result": {"message_id": 123}}
            with patch("claude_telegram.adapters.telegram.handlers.telegram.delete_message", new_callable=AsyncMock):
                with patch("claude_telegram.adapters.telegram.handlers.send_response", new_callable=AsyncMock) as mock_chunked:
                    await run_claude("Hello", "12345", bot, continue_session=False)
                    mock_runner.run.assert_called_once()
                    mock_chunked.assert_called_once_with("Claude response", "12345", session_name="test", api_url=bot.api_url, message_thread_id=None)


@pytest.mark.asyncio
async def test_run_claude_error():
    """Test Claude run with error."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = False
    mock_runner.run = AsyncMock(side_effect=Exception("Test error"))
    mock_runner.short_name = "test"
    with patch("claude_telegram.adapters.telegram.handlers.get_runner", return_value=mock_runner):
        with patch("claude_telegram.adapters.telegram.handlers.telegram.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"result": {"message_id": 123}}
            with patch("claude_telegram.adapters.telegram.handlers.telegram.delete_message", new_callable=AsyncMock):
                await run_claude("Hello", "12345", bot, continue_session=False)
                # Should have sent error message
                calls = mock_send.call_args_list
                assert any("Error" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_send_response_short():
    """Test send_response with short text."""
    with patch("claude_telegram.adapters.telegram.outbound.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await send_response("Short text", "12345")
        mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_send_response_empty():
    """Test send_response with empty text."""
    with patch("claude_telegram.adapters.telegram.outbound.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await send_response("", "12345")
        mock_send.assert_called_once()
        assert "no output" in mock_send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_send_response_long():
    """Test send_response with long text requiring multiple messages."""
    # Text with newlines to test chunking (split_text breaks at newlines)
    long_text = ("x" * 3000 + "\n") * 3  # ~9000 chars with newlines
    with patch("claude_telegram.adapters.telegram.outbound.telegram.send_message", new_callable=AsyncMock) as mock_send:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await send_response(long_text, "12345")
            assert mock_send.call_count >= 2  # Should split into multiple chunks


class FakeNotifications:
    """Records what producers publish instead of calling a chat API."""

    def __init__(self, ref=None):
        self.events = []
        self.opened = []
        self.ref = ref

    async def publish(self, event):
        self.events.append(event)
        return []

    async def open_conversation(self, event):
        self.opened.append(event)
        return self.ref


def test_notify_completed_publishes_dev_event():
    bot = _make_dev_bot()
    fake = FakeNotifications()
    with patch.object(state, "bots", {"dev": bot}), patch.object(state, "notifications", fake):
        response = client.post("/notify/completed", json={
            "summary": "Tout est **fait**", "working_dir": "/home/x/projet", "session_id": "abc"})
    assert response.json()["ok"] is True
    [event] = fake.events
    assert event.type == "dev.completed"
    assert "`projet`" in event.body and "Tout est **fait**" in event.body
    assert [(a.label, a.data) for a in event.actions] == [("Continue ➜", "resume:abc")]
    assert state.resume_working_dirs["abc"] == "/home/x/projet"


def test_notify_waiting():
    bot = _make_dev_bot()
    fake = FakeNotifications()
    with patch.object(state, "bots", {"dev": bot}), patch.object(state, "notifications", fake):
        response = client.post("/notify/waiting")
    assert response.json()["ok"] is True
    assert fake.events[0].type == "dev.waiting"


def test_notify_custom():
    bot = _make_dev_bot()
    fake = FakeNotifications()
    with patch.object(state, "bots", {"dev": bot}), patch.object(state, "notifications", fake):
        response = client.post("/notify/custom_event")
    assert response.status_code == 200
    assert fake.events[0].type == "dev.custom_event"


def test_notify_fitness_opens_conversation_and_enqueues():
    from claude_telegram.ports import ConversationRef

    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    ref = ConversationRef("telegram", "999", thread_id=77, bot="gtd")
    fake = FakeNotifications(ref=ref)
    queue = MagicMock()
    queue.enqueue = AsyncMock(return_value=1)
    with patch.object(state, "bots", {"gtd": gtd}), patch.object(state, "notifications", fake), \
            patch.object(state, "gtd_queue", queue):
        response = client.post("/notify/fitness-completed", json={"summary": "Squats 3x10"})
    assert response.json()["ok"] is True
    assert fake.opened[0].type == "fitness"
    item = queue.enqueue.await_args.args[0]
    assert item.conversation == ref and item.thread_id == 77
    assert "Squats 3x10" in item.prompt


def _pipeline_run(tmp_path, reminder_type, *, run):
    import asyncio
    from claude_telegram import main

    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    fake = FakeNotifications()
    with patch.object(state, "notifications", fake), patch("subprocess.run", run), \
            patch.object(main.settings, "gtd_working_dir", str(tmp_path)):
        asyncio.run(main._process_pipeline_cron(reminder_type, gtd))
    return fake


def test_pipeline_cron_output_publishes_event(tmp_path):
    from types import SimpleNamespace

    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="Briefing du jour\n", stderr=""))
    fake = _pipeline_run(tmp_path, "morning", run=run)
    [event] = fake.events
    assert (event.type, event.severity, event.title) == ("briefing_morning", "normal", "Cron: morning")
    assert event.body == "Briefing du jour"


def test_pipeline_cron_severity_marker(tmp_path):
    from types import SimpleNamespace

    run = MagicMock(return_value=SimpleNamespace(
        returncode=0, stdout="<!-- severity: urgent -->\nRDV annulé", stderr=""))
    fake = _pipeline_run(tmp_path, "whatsapp", run=run)
    assert (fake.events[0].type, fake.events[0].severity, fake.events[0].body) == (
        "whatsapp_triage", "urgent", "RDV annulé")


def test_pipeline_cron_ok_is_silent(tmp_path):
    from types import SimpleNamespace

    fake = _pipeline_run(tmp_path, "zulip", run=MagicMock(
        return_value=SimpleNamespace(returncode=0, stdout="OK", stderr="")))
    assert fake.events == []


def test_pipeline_cron_failure_is_normal_event(tmp_path):
    from types import SimpleNamespace

    fake = _pipeline_run(tmp_path, "evening", run=MagicMock(
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="Traceback boom")))
    [event] = fake.events
    assert event.severity == "normal" and "❌ Pipeline evening" in event.body and "boom" in event.body


def test_pipeline_cron_timeout_is_urgent(tmp_path):
    import subprocess

    fake = _pipeline_run(tmp_path, "morning", run=MagicMock(
        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=600)))
    [event] = fake.events
    assert event.severity == "urgent" and "timeout" in event.body


def test_channels_inject_requires_secret():
    from claude_telegram import main

    fake = FakeNotifications()
    with patch.object(state, "notifications", fake), patch.object(main.settings, "webhook_secret", "s3cret"), \
            patch.object(main, "INJECT_HOSTS", {"testclient"}):
        denied = client.post("/channels/inject", json={"event": {"type": "test"}})
        wrong = client.post("/channels/inject", json={"event": {"type": "test"}},
                            headers={"X-Webhook-Secret": "nope"})
    assert denied.status_code == 401 and wrong.status_code == 401
    assert fake.events == []


def test_channels_inject_refuses_remote_hosts():
    from claude_telegram import main

    fake = FakeNotifications()
    with patch.object(state, "notifications", fake), patch.object(main.settings, "webhook_secret", "s3cret"):
        response = client.post("/channels/inject", json={"event": {"type": "test"}},
                               headers={"X-Webhook-Secret": "s3cret"})
    assert response.status_code == 403 and fake.events == []


def test_channels_inject_publishes_event():
    from claude_telegram import main

    fake = FakeNotifications()
    with patch.object(state, "notifications", fake), patch.object(main.settings, "webhook_secret", "s3cret"), \
            patch.object(main, "INJECT_HOSTS", {"testclient"}):
        response = client.post("/channels/inject", headers={"X-Webhook-Secret": "s3cret"}, json={
            "event": {"type": "test", "severity": "normal", "title": "Cron: test", "body": "Test étape 1"}})
    assert response.status_code == 200
    [event] = fake.events
    assert (event.type, event.severity, event.title, event.body) == ("test", "normal", "Cron: test", "Test étape 1")


class FakeInbound:
    def __init__(self, seen=()):
        self._seen = {int(i) for i in seen}
        self.saved = 0

    def seen(self, message_id):
        try:
            return int(message_id) in self._seen
        except (TypeError, ValueError):
            return False

    def mark_seen(self, message_id):
        self._seen.add(int(message_id))

    def save_state(self):
        self.saved += 1


def _zulip_post(tmp_path, message_id, inbound):
    from claude_telegram import main

    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    pipeline, fallback = AsyncMock(), AsyncMock()
    with patch.dict("os.environ", {"ZULIP_WEBHOOK_TOKEN": "tok"}), \
            patch.object(main.settings, "gtd_working_dir", str(tmp_path)), \
            patch.object(state, "bots", {"gtd": gtd}), patch.object(state, "zulip_inbound", inbound), \
            patch.object(main, "_process_pipeline_cron", pipeline), \
            patch.object(main, "_delayed_zulip_fallback", fallback):
        response = client.post("/webhook/zulip", json={
            "token": "tok", "data": "@**Agent** salut",
            "message": {"id": message_id, "content": "@**Agent** salut", "sender_email": "a@b"}})
    return response, pipeline, fallback


def test_zulip_webhook_ignores_message_already_seen_by_inbound(tmp_path):
    response, pipeline, fallback = _zulip_post(tmp_path, 42, FakeInbound(seen=[42]))
    assert response.json() == {}
    assert not (tmp_path / "data" / "zulip-pending" / "42.json").exists()
    pipeline.assert_not_called()
    fallback.assert_not_called()


def test_zulip_webhook_defers_to_inbound_with_delayed_fallback(tmp_path):
    response, pipeline, fallback = _zulip_post(tmp_path, 43, FakeInbound())
    assert response.json() == {}
    assert (tmp_path / "data" / "zulip-pending" / "43.json").exists()
    pipeline.assert_not_called()
    assert fallback.call_args.args[0] == 43


def test_zulip_webhook_without_inbound_runs_pipeline(tmp_path):
    response, pipeline, fallback = _zulip_post(tmp_path, 44, None)
    assert (tmp_path / "data" / "zulip-pending" / "44.json").exists()
    assert pipeline.call_args.args[0] == "zulip"
    fallback.assert_not_called()


def _fallback_run(tmp_path, message_id, inbound):
    import asyncio
    from claude_telegram import main

    pending = tmp_path / "data" / "zulip-pending"
    pending.mkdir(parents=True)
    (pending / f"{message_id}.json").write_text("{}")
    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    pipeline = AsyncMock()
    with patch.object(main.settings, "gtd_working_dir", str(tmp_path)), \
            patch.object(state, "zulip_inbound", inbound), patch.object(main, "_process_pipeline_cron", pipeline):
        asyncio.run(main._delayed_zulip_fallback(message_id, gtd, delay=0))
    return pending / f"{message_id}.json", pipeline


def test_delayed_fallback_drops_payload_handled_by_inbound(tmp_path):
    path, pipeline = _fallback_run(tmp_path, 50, FakeInbound(seen=[50]))
    assert not path.exists()
    pipeline.assert_not_called()


def test_delayed_fallback_runs_pipeline_when_inbound_missed_it(tmp_path):
    inbound = FakeInbound()
    path, pipeline = _fallback_run(tmp_path, 51, inbound)
    assert pipeline.call_args.args[0] == "zulip"
    assert inbound.seen(51) and inbound.saved == 1


def test_cron_zulip_purges_payloads_seen_by_inbound(tmp_path):
    from claude_telegram import main

    pending = tmp_path / "data" / "zulip-pending"
    pending.mkdir(parents=True)
    (pending / "60.json").write_text("{}")
    (pending / "61.json").write_text("{}")
    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    pipeline = AsyncMock()
    with patch.object(main.settings, "gtd_working_dir", str(tmp_path)), \
            patch.object(state, "bots", {"gtd": gtd}), patch.object(state, "zulip_inbound", FakeInbound(seen=[60])), \
            patch.object(main, "_process_pipeline_cron", pipeline):
        response = client.post("/cron/zulip")
    assert response.json()["mode"] == "pipeline"
    assert not (pending / "60.json").exists() and (pending / "61.json").exists()
    assert pipeline.call_args.args[0] == "zulip"


def test_process_email_enqueues_proposal_prompt_without_backend_filters():
    """Noise is filtered by the gate: the backend must queue every email it gets,
    otherwise the email would stay stuck in `analysing` in the webapp."""
    import asyncio
    from claude_telegram import main

    gtd = BotConfig(name="gtd", token="t", chat_id="999", use_queue=True)
    queue = MagicMock()
    queue.enqueue = AsyncMock(return_value=1)
    data = {"messageId": "m1", "threadId": "t1", "from": "GitHub <notifications@github.com>",
            "subject": "Lifen : documents reçus"}
    with patch.object(state, "gtd_queue", queue):
        asyncio.run(main._process_email(data, gtd))

    item = queue.enqueue.await_args.args[0]
    assert "python3 -m scripts.inbox.cli propose --message-id m1" in item.prompt
    assert item.source == "email" and item.event_type == "email_triage"
    assert item.thread_id is None and item.new_session is True
    assert item.metadata == {"subject": "Lifen : documents reçus", "from": "GitHub <notifications@github.com>"}
