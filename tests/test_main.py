"""Tests for FastAPI main application."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Must patch before importing app
with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "test_token",
    "TELEGRAM_CHAT_ID": "12345",
}):
    from claude_telegram.main import app, handle_message, handle_command, run_claude, send_response
    from claude_telegram.bots import BotConfig


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
    import claude_telegram.main as main_mod
    bot = _make_dev_bot()
    with patch.object(main_mod, "bots", {"dev": bot}):
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
    with patch("claude_telegram.main.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_message(msg, bot)
        mock_run.assert_called_once_with("Hello Claude", "12345", bot, continue_session=False, thread_id=42, new_session=False)


@pytest.mark.asyncio
async def test_handle_message_unauthorized(unauthorized_message):
    """Test handling unauthorized message."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.run_claude", new_callable=AsyncMock) as mock_run:
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
    with patch("claude_telegram.main.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_message(message, bot)
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_handle_command_start():
    """Test /start command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/start", "12345", bot)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "Commands" in call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_continue():
    """Test /c command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_command("/c fix the bug", "12345", bot)
        mock_run.assert_called_once_with("fix the bug", "12345", bot, continue_session=True, thread_id=None)


@pytest.mark.asyncio
async def test_handle_command_continue_alias():
    """Test /continue command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.run_claude", new_callable=AsyncMock) as mock_run:
        await handle_command("/continue do something", "12345", bot)
        mock_run.assert_called_once_with("do something", "12345", bot, continue_session=True, thread_id=None)


@pytest.mark.asyncio
async def test_handle_command_continue_no_args():
    """Test /c command without arguments."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
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
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock):
            with patch("claude_telegram.main.send_response", new_callable=AsyncMock) as mock_chunked:
                await handle_command("/compact", "12345", bot)
                mock_runner.compact.assert_called_once()
                mock_chunked.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_compact_while_busy():
    """Test /compact when Claude is running."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.is_running = True
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/compact", "12345", bot)
            assert "busy" in mock_send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_handle_command_cancel():
    """Test /cancel command."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.cancel = AsyncMock(return_value=True)
    mock_runner.short_name = "test"
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/cancel", "12345", bot)
            mock_runner.cancel.assert_called_once()
            assert "Cancelled" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_cancel_nothing():
    """Test /cancel when nothing is running."""
    bot = _make_dev_bot()
    mock_runner = MagicMock()
    mock_runner.cancel = AsyncMock(return_value=False)
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
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
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/status", "12345", bot)
            assert "Running" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_unknown():
    """Test unknown command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
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
    with patch("claude_telegram.main.sessions") as mock_sessions:
        mock_sessions.switch_session = MagicMock(return_value=mock_session)
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dir /path/to/myproject", "12345", bot)
            mock_sessions.switch_session.assert_called_once_with("/path/to/myproject")
            assert "Switched" in mock_send.call_args[0][0]
            assert "myproject" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_command_dir_no_args():
    """Test /dir command without arguments shows directory browser."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await handle_command("/dir", "12345", bot)
        msg = mock_send.call_args[0][0]
        assert "Current" in msg


@pytest.mark.asyncio
async def test_handle_command_dirs():
    """Test /dirs command."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.sessions") as mock_sessions:
        mock_sessions.list_dirs = MagicMock(return_value=[
            ("/path/to/project1", 2),
            ("/path/to/project2", 1),
        ])
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dirs", "12345", bot)
            message = mock_send.call_args[0][0]
            assert "Active Directories" in message
            assert "project1" in message
            assert "project2" in message


@pytest.mark.asyncio
async def test_handle_command_dirs_empty():
    """Test /dirs command with no sessions."""
    bot = _make_dev_bot()
    with patch("claude_telegram.main.sessions") as mock_sessions:
        mock_sessions.list_dirs = MagicMock(return_value=[])
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            await handle_command("/dirs", "12345", bot)
            assert "No active sessions" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_callback_dir_switch():
    """Test callback for directory switching."""
    from claude_telegram.main import handle_callback
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
    with patch("claude_telegram.main.sessions") as mock_sessions:
        mock_sessions.switch_session = MagicMock(return_value=mock_session)
        with patch("claude_telegram.main.telegram.answer_callback", new_callable=AsyncMock):
            with patch("claude_telegram.main.telegram.edit_message", new_callable=AsyncMock) as mock_edit:
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
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
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
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"result": {"message_id": 123}}
            with patch("claude_telegram.main.telegram.delete_message", new_callable=AsyncMock):
                with patch("claude_telegram.main.send_response", new_callable=AsyncMock) as mock_chunked:
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
    with patch("claude_telegram.main.get_runner", return_value=mock_runner):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"result": {"message_id": 123}}
            with patch("claude_telegram.main.telegram.delete_message", new_callable=AsyncMock):
                await run_claude("Hello", "12345", bot, continue_session=False)
                # Should have sent error message
                calls = mock_send.call_args_list
                assert any("Error" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_send_response_short():
    """Test send_response with short text."""
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await send_response("Short text", "12345")
        mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_send_response_empty():
    """Test send_response with empty text."""
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
        await send_response("", "12345")
        mock_send.assert_called_once()
        assert "no output" in mock_send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_send_response_long():
    """Test send_response with long text requiring multiple messages."""
    # Text with newlines to test chunking (split_text breaks at newlines)
    long_text = ("x" * 3000 + "\n") * 3  # ~9000 chars with newlines
    with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock) as mock_send:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await send_response(long_text, "12345")
            assert mock_send.call_count >= 2  # Should split into multiple chunks


def test_notify_completed():
    """Test notification endpoint for completed."""
    import claude_telegram.main as main_mod
    bot = _make_dev_bot()
    with patch.object(main_mod, "bots", {"dev": bot}):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock):
            response = client.post("/notify/completed")
            assert response.status_code == 200
            assert response.json()["ok"] is True


def test_notify_waiting():
    """Test notification endpoint for waiting."""
    import claude_telegram.main as main_mod
    bot = _make_dev_bot()
    with patch.object(main_mod, "bots", {"dev": bot}):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock):
            response = client.post("/notify/waiting")
            assert response.status_code == 200
            assert response.json()["ok"] is True


def test_notify_custom():
    """Test notification endpoint for custom event."""
    import claude_telegram.main as main_mod
    bot = _make_dev_bot()
    with patch.object(main_mod, "bots", {"dev": bot}):
        with patch("claude_telegram.main.telegram.send_message", new_callable=AsyncMock):
            response = client.post("/notify/custom_event")
            assert response.status_code == 200
