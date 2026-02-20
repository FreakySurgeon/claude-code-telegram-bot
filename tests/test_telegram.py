"""Tests for Telegram service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

# Import after setting env vars in conftest
from claude_telegram import telegram


@pytest.mark.asyncio
async def test_send_message_success(mock_httpx):
    """Test successful message sending."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.send_message("Hello", chat_id="12345")

    assert result["ok"] is True
    mock_httpx.post.assert_called_once()
    call_args = mock_httpx.post.call_args
    assert "sendMessage" in call_args[0][0]
    assert call_args[1]["json"]["text"] == "Hello"
    assert call_args[1]["json"]["chat_id"] == "12345"


@pytest.mark.asyncio
async def test_send_message_with_reply_markup(mock_httpx):
    """Test message with inline keyboard."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    markup = {"inline_keyboard": [[{"text": "Button", "callback_data": "test"}]]}
    await telegram.send_message("Choose:", reply_markup=markup)

    call_args = mock_httpx.post.call_args
    assert call_args[1]["json"]["reply_markup"] == markup


@pytest.mark.asyncio
async def test_send_message_http_error(mock_httpx):
    """Test handling of HTTP errors."""
    mock_httpx.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "Error", request=MagicMock(), response=MagicMock(status_code=400)
    ))

    with pytest.raises(httpx.HTTPStatusError):
        await telegram.send_message("Hello")


@pytest.mark.asyncio
async def test_edit_message_success(mock_httpx):
    """Test successful message editing."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.edit_message(123, "Updated text", chat_id="12345")

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert "editMessageText" in call_args[0][0]
    assert call_args[1]["json"]["message_id"] == 123
    assert call_args[1]["json"]["text"] == "Updated text"


@pytest.mark.asyncio
async def test_set_webhook(mock_httpx):
    """Test webhook setup."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.set_webhook("https://example.com/webhook")

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert "setWebhook" in call_args[0][0]
    assert call_args[1]["json"]["url"] == "https://example.com/webhook"


@pytest.mark.asyncio
async def test_delete_webhook(mock_httpx):
    """Test webhook deletion."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.delete_webhook()

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert "deleteWebhook" in call_args[0][0]


def test_is_authorized_valid():
    """Test authorization with valid chat ID."""
    assert telegram.is_authorized(12345) is True
    assert telegram.is_authorized("12345") is True


def test_is_authorized_invalid():
    """Test authorization with invalid chat ID."""
    assert telegram.is_authorized(99999) is False
    assert telegram.is_authorized("invalid") is False


# --- Forum Topics / message_thread_id tests ---


@pytest.mark.asyncio
async def test_send_message_with_thread_id(mock_httpx):
    """Test send_message passes message_thread_id when provided."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 456}}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.send_message(
        "Hello topic", chat_id="12345", message_thread_id=99
    )

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert call_args[1]["json"]["message_thread_id"] == 99


@pytest.mark.asyncio
async def test_send_message_without_thread_id(mock_httpx):
    """Test send_message does NOT include message_thread_id when None."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 789}}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.send_message("Hello", chat_id="12345")

    call_args = mock_httpx.post.call_args
    assert "message_thread_id" not in call_args[1]["json"]


@pytest.mark.asyncio
async def test_edit_message_with_thread_id(mock_httpx):
    """Test edit_message passes message_thread_id when provided."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.edit_message(
        123, "Updated text", chat_id="12345", message_thread_id=42
    )

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert call_args[1]["json"]["message_thread_id"] == 42


@pytest.mark.asyncio
async def test_edit_message_without_thread_id(mock_httpx):
    """Test edit_message does NOT include message_thread_id when None."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    await telegram.edit_message(123, "Updated", chat_id="12345")

    call_args = mock_httpx.post.call_args
    assert "message_thread_id" not in call_args[1]["json"]


@pytest.mark.asyncio
async def test_create_forum_topic(mock_httpx):
    """Test create_forum_topic posts to correct endpoint with name."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "result": {"message_thread_id": 100, "name": "My Topic"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.create_forum_topic("12345", "My Topic")

    assert result["ok"] is True
    assert result["result"]["message_thread_id"] == 100
    call_args = mock_httpx.post.call_args
    assert "createForumTopic" in call_args[0][0]
    assert call_args[1]["json"]["chat_id"] == "12345"
    assert call_args[1]["json"]["name"] == "My Topic"


@pytest.mark.asyncio
async def test_create_forum_topic_truncates_name(mock_httpx):
    """Test create_forum_topic truncates name to 128 chars."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_thread_id": 101}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    long_name = "A" * 200
    await telegram.create_forum_topic("12345", long_name)

    call_args = mock_httpx.post.call_args
    assert len(call_args[1]["json"]["name"]) == 128


@pytest.mark.asyncio
async def test_create_forum_topic_custom_api_url(mock_httpx):
    """Test create_forum_topic uses custom api_url."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_thread_id": 102}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    await telegram.create_forum_topic("12345", "Topic", api_url="https://custom.api/botXYZ")

    call_args = mock_httpx.post.call_args
    assert call_args[0][0] == "https://custom.api/botXYZ/createForumTopic"


@pytest.mark.asyncio
async def test_edit_forum_topic(mock_httpx):
    """Test edit_forum_topic posts to correct endpoint."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.edit_forum_topic("12345", 100, "Renamed Topic")

    assert result["ok"] is True
    call_args = mock_httpx.post.call_args
    assert "editForumTopic" in call_args[0][0]
    assert call_args[1]["json"]["chat_id"] == "12345"
    assert call_args[1]["json"]["message_thread_id"] == 100
    assert call_args[1]["json"]["name"] == "Renamed Topic"


@pytest.mark.asyncio
async def test_edit_forum_topic_truncates_name(mock_httpx):
    """Test edit_forum_topic truncates name to 128 chars."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    long_name = "B" * 300
    await telegram.edit_forum_topic("12345", 100, long_name)

    call_args = mock_httpx.post.call_args
    assert len(call_args[1]["json"]["name"]) == 128


@pytest.mark.asyncio
async def test_get_chat(mock_httpx):
    """Test get_chat posts to correct endpoint."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "result": {
            "id": -1001234567890,
            "type": "supergroup",
            "is_forum": True,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    result = await telegram.get_chat("12345")

    assert result["ok"] is True
    assert result["result"]["is_forum"] is True
    call_args = mock_httpx.post.call_args
    assert "getChat" in call_args[0][0]
    assert call_args[1]["json"]["chat_id"] == "12345"


@pytest.mark.asyncio
async def test_get_chat_custom_api_url(mock_httpx):
    """Test get_chat uses custom api_url."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"id": 123}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    await telegram.get_chat("12345", api_url="https://custom.api/botXYZ")

    call_args = mock_httpx.post.call_args
    assert call_args[0][0] == "https://custom.api/botXYZ/getChat"
