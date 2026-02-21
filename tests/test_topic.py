"""Tests for topic naming logic."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from claude_telegram.topic import (
    generate_provisional_name,
    extract_title_from_response,
    generate_title_fallback,
    format_topic_name,
    working_dir_name,
    MAX_TOPIC_NAME,
)

FAKE_DATE = "15/02"
FAKE_NOW = datetime(2026, 2, 15, 10, 30, 0)


@pytest.fixture(autouse=True)
def _mock_datetime():
    """Mock datetime.now() to return a fixed date for all tests."""
    with patch("claude_telegram.topic.datetime") as mock_dt:
        mock_dt.now.return_value = FAKE_NOW
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        yield mock_dt


# --- generate_provisional_name ---


class TestGenerateProvisionalName:
    """Test provisional topic name generation."""

    def test_short_message(self):
        """Short message is kept in full."""
        result = generate_provisional_name("Bonjour")
        assert result == f"{FAKE_DATE} - Bonjour"

    def test_long_message_truncated(self):
        """Long message is truncated with ellipsis."""
        long_msg = "A" * 200
        result = generate_provisional_name(long_msg)
        assert len(result) <= MAX_TOPIC_NAME
        assert result.endswith("...")

    def test_dir_name_prefix(self):
        """dir_name adds [name] prefix."""
        result = generate_provisional_name("Test", dir_name="personal-org")
        assert result.startswith(f"[personal-org] {FAKE_DATE} - ")
        assert "Test" in result

    def test_no_dir_name_no_prefix(self):
        """No dir_name has no bracket prefix."""
        result = generate_provisional_name("Test")
        assert not result.startswith("[")
        assert result.startswith(f"{FAKE_DATE} - ")

    def test_command_prefix_stripped(self):
        """Leading /command is stripped from message."""
        result = generate_provisional_name("/new Fix the bug")
        assert "Fix the bug" in result
        assert "/new" not in result

    def test_command_only_uses_default(self):
        """Bare command with no text uses default name."""
        result = generate_provisional_name("/new")
        assert "Nouvelle conversation" in result

    def test_empty_message_uses_default(self):
        """Empty message uses default name."""
        result = generate_provisional_name("")
        assert "Nouvelle conversation" in result

    def test_whitespace_only_uses_default(self):
        """Whitespace-only message uses default name."""
        result = generate_provisional_name("   ")
        assert "Nouvelle conversation" in result

    def test_result_within_limit(self):
        """Result never exceeds Telegram limit."""
        long_msg = "B" * 300
        result = generate_provisional_name(long_msg, dir_name="personal-org")
        assert len(result) <= MAX_TOPIC_NAME


# --- extract_title_from_response ---


class TestExtractTitleFromResponse:
    """Test title extraction from HTML comment."""

    def test_title_found(self):
        """Title comment is extracted and text cleaned."""
        response = "<!-- title: Résumé du budget -->\nVoici le résumé..."
        cleaned, title = extract_title_from_response(response)
        assert title == "Résumé du budget"
        assert "<!-- title:" not in cleaned
        assert "Voici le résumé..." in cleaned

    def test_title_not_found(self):
        """No title comment returns original text and None."""
        response = "Just a normal response without any title."
        cleaned, title = extract_title_from_response(response)
        assert title is None
        assert cleaned == response

    def test_title_in_middle_of_text(self):
        """Title comment in the middle of text is extracted."""
        response = "Before <!-- title: Mon titre --> After"
        cleaned, title = extract_title_from_response(response)
        assert title == "Mon titre"
        assert "Before" in cleaned
        assert "After" in cleaned
        assert "<!-- title:" not in cleaned

    def test_title_with_extra_whitespace(self):
        """Whitespace around title is stripped."""
        response = "<!--  title:   Spaces everywhere   -->"
        cleaned, title = extract_title_from_response(response)
        assert title == "Spaces everywhere"
        assert cleaned == ""

    def test_empty_response(self):
        """Empty response returns empty string and None."""
        cleaned, title = extract_title_from_response("")
        assert title is None
        assert cleaned == ""


# --- generate_title_fallback ---


class TestGenerateTitleFallback:
    """Test Ollama-based title fallback generation."""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Successful Ollama call returns cleaned title."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "Planification budget mensuel"}

        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.return_value = mock_response

            result = await generate_title_fallback("Aide-moi avec le budget", "Voici un plan...")
            assert result == "Planification budget mensuel"

    @pytest.mark.asyncio
    async def test_strips_quotes_and_punctuation(self):
        """Quotes and trailing punctuation are cleaned."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": '"Budget mensuel."'}

        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.return_value = mock_response

            result = await generate_title_fallback("Test", "Response")
            assert result == "Budget mensuel"

    @pytest.mark.asyncio
    async def test_network_error_returns_fallback(self):
        """Network error falls back to truncated message."""
        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.side_effect = httpx.ConnectError("Connection refused")

            result = await generate_title_fallback("Mon message original", "Response")
            assert result == "Mon message original"

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(self):
        """Timeout falls back to truncated message."""
        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.side_effect = httpx.ReadTimeout("Timeout")

            result = await generate_title_fallback("/new Fix the bug", "Response")
            assert result == "Fix the bug"

    @pytest.mark.asyncio
    async def test_empty_ollama_response_returns_fallback(self):
        """Empty Ollama response falls back to message."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": ""}

        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.return_value = mock_response

            result = await generate_title_fallback("Fallback message", "Response")
            assert result == "Fallback message"

    @pytest.mark.asyncio
    async def test_long_fallback_truncated_to_50_chars(self):
        """Fallback message longer than 50 chars is truncated with ellipsis."""
        long_message = "A" * 80
        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.side_effect = httpx.ConnectError("Connection refused")

            result = await generate_title_fallback(long_message, "Response")
            assert result == "A" * 50 + "..."
            assert len(result) == 53

    @pytest.mark.asyncio
    async def test_result_within_limit(self):
        """Result never exceeds Telegram limit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "A" * 200}

        with patch("claude_telegram.topic.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = client
            mock_client_cls.return_value.__aexit__.return_value = None
            client.post.return_value = mock_response

            result = await generate_title_fallback("Test", "Response")
            assert len(result) <= MAX_TOPIC_NAME


# --- format_topic_name ---


class TestFormatTopicName:
    """Test final topic name formatting."""

    def test_basic_format(self):
        """Basic title is formatted with date."""
        result = format_topic_name("Mon sujet")
        assert result == f"{FAKE_DATE} - Mon sujet"

    def test_dir_name_prefix(self):
        """dir_name adds [name] prefix."""
        result = format_topic_name("Mon sujet", dir_name="personal-org")
        assert result == f"[personal-org] {FAKE_DATE} - Mon sujet"

    def test_long_title_truncated(self):
        """Long title is truncated with ellipsis."""
        long_title = "X" * 200
        result = format_topic_name(long_title)
        assert len(result) <= MAX_TOPIC_NAME
        assert result.endswith("...")

    def test_long_title_dir_name_truncated(self):
        """Long title with dir_name prefix is still within limit."""
        long_title = "Y" * 200
        result = format_topic_name(long_title, dir_name="personal-org")
        assert len(result) <= MAX_TOPIC_NAME
        assert result.startswith("[personal-org]")
        assert result.endswith("...")

    def test_exact_limit(self):
        """Title exactly at the limit is not truncated."""
        prefix = f"{FAKE_DATE} - "
        exact_title = "Z" * (MAX_TOPIC_NAME - len(prefix))
        result = format_topic_name(exact_title)
        assert len(result) == MAX_TOPIC_NAME
        assert "..." not in result


# --- working_dir_name ---


class TestWorkingDirName:
    """Test working directory name extraction."""

    def test_full_path(self):
        assert working_dir_name("/home/thomas/projects/personal-org") == "personal-org"

    def test_home_dir(self):
        assert working_dir_name("/home/thomas") == "thomas"

    def test_none(self):
        assert working_dir_name(None) is None

    def test_empty(self):
        assert working_dir_name("") is None
