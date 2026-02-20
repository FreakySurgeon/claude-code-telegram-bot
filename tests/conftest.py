"""Pytest configuration and fixtures."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment before importing app modules
os.environ["TELEGRAM_BOT_TOKEN"] = "test_token_123"
os.environ["TELEGRAM_CHAT_ID"] = "12345"
os.environ["CLAUDE_CLI_PATH"] = "claude"
os.environ["CLAUDE_WORKING_DIR"] = "/tmp/test"
# GTD bot env vars (optional, can be empty)
os.environ.setdefault("GTD_BOT_TOKEN", "")
os.environ.setdefault("GTD_CHAT_ID", "")
os.environ.setdefault("GTD_WORKING_DIR", "")
os.environ.setdefault("GTD_PROMPT_PATH", "")
os.environ.setdefault("GTD_MCP_CONFIG", "")
os.environ.setdefault("MISTRAL_API_KEY", "")
os.environ.setdefault("WEBHOOK_SECRET", "")


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("claude_telegram.config.settings") as mock:
        mock.telegram_bot_token = "test_token_123"
        mock.telegram_chat_id = "12345"
        mock.claude_cli_path = "claude"
        mock.claude_working_dir = "/tmp/test"
        mock.host = "0.0.0.0"
        mock.port = 8000
        mock.webhook_path = "/webhook"
        mock.webhook_url = None
        mock.gtd_bot_token = None
        mock.gtd_chat_id = None
        mock.gtd_working_dir = None
        mock.gtd_prompt_path = None
        mock.gtd_mcp_config = None
        mock.mistral_api_key = None
        mock.webhook_secret = None
        yield mock


@pytest.fixture
def mock_httpx():
    """Mock httpx for Telegram API calls."""
    with patch("claude_telegram.telegram.httpx.AsyncClient") as mock:
        client = AsyncMock()
        mock.return_value.__aenter__.return_value = client
        mock.return_value.__aexit__.return_value = None
        yield client


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for Claude runner."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.stdout = AsyncMock()
        process.wait = AsyncMock(return_value=0)
        mock.return_value = process
        yield mock, process


@pytest.fixture
def dev_bot():
    """Create a dev BotConfig for testing."""
    from claude_telegram.bots import BotConfig
    return BotConfig(
        name="dev",
        token="test_token_123",
        chat_id="12345",
        use_queue=False,
        commands_whitelist=[
            "/start", "/help", "/c", "/continue", "/new", "/dir", "/dirs",
            "/repos", "/rmdir", "/compact", "/cancel", "/status",
        ],
    )


@pytest.fixture
def gtd_bot():
    """Create a GTD BotConfig for testing."""
    from claude_telegram.bots import BotConfig
    return BotConfig(
        name="gtd",
        token="gtd_test_token",
        chat_id="67890",
        fixed_working_dir="/tmp/gtd",
        use_queue=True,
        commands_whitelist=[
            "/start", "/help", "/new", "/compact", "/cancel", "/status",
        ],
    )


@pytest.fixture
def authorized_message():
    """Sample authorized Telegram message."""
    return {
        "message": {
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 12345, "first_name": "Test"},
            "text": "Hello Claude",
        }
    }


@pytest.fixture
def unauthorized_message():
    """Sample unauthorized Telegram message."""
    return {
        "message": {
            "message_id": 1,
            "chat": {"id": 99999, "type": "private"},
            "from": {"id": 99999, "first_name": "Hacker"},
            "text": "Hello Claude",
        }
    }


@pytest.fixture
def continue_command():
    """Sample continue command message."""
    return {
        "message": {
            "message_id": 2,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 12345, "first_name": "Test"},
            "text": "/c fix the bug",
        }
    }


@pytest.fixture
def compact_command():
    """Sample compact command message."""
    return {
        "message": {
            "message_id": 3,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 12345, "first_name": "Test"},
            "text": "/compact",
        }
    }
