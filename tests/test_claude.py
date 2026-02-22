"""Tests for Claude runner."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from claude_telegram.claude import ClaudeRunner, ClaudeResult, PermissionDenial


@pytest.fixture
def runner():
    """Create a fresh runner for each test."""
    return ClaudeRunner()


@pytest.fixture
def mock_process():
    """Create a mock subprocess."""
    process = AsyncMock()
    process.wait = AsyncMock(return_value=0)
    process.terminate = MagicMock()
    return process


async def async_iter(items):
    """Helper to create async iterator."""
    for item in items:
        yield item


def make_stream_json(result_text: str, permission_denials: list = None):
    """Create mock stream-json output."""
    events = [
        # Init event
        json.dumps({"type": "system", "subtype": "init", "session_id": "test-session"}).encode() + b"\n",
        # Assistant response
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": result_text}]}
        }).encode() + b"\n",
        # Result event
        json.dumps({
            "type": "result",
            "result": result_text,
            "session_id": "test-session",
            "permission_denials": permission_denials or []
        }).encode() + b"\n",
    ]
    return events


@pytest.mark.asyncio
async def test_run_basic_message(runner, mock_process):
    """Test running Claude with a basic message."""
    mock_process.stdout = async_iter(make_stream_json("Hello from Claude!"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await runner.run("Hello")

        assert isinstance(result, ClaudeResult)
        assert "Hello from Claude!" in result.text
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0]
        assert "--print" in call_args
        assert "--output-format" in call_args
        assert "stream-json" in call_args


@pytest.mark.asyncio
async def test_run_with_continue(runner, mock_process):
    """Test running Claude with --continue flag."""
    mock_process.stdout = async_iter(make_stream_json("Continued response"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await runner.run("Continue this", continue_session=True)

        assert "Continued response" in result.text
        call_args = mock_exec.call_args[0]
        assert "--continue" in call_args


@pytest.mark.asyncio
async def test_run_without_continue(runner, mock_process):
    """Test running Claude without --continue flag."""
    mock_process.stdout = async_iter(make_stream_json("New session"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await runner.run("New message", continue_session=False)

        call_args = mock_exec.call_args[0]
        assert "--continue" not in call_args


@pytest.mark.asyncio
async def test_run_with_callback(runner, mock_process):
    """Test running Claude with output callback."""
    # Create events with multiple text chunks
    events = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "test"}).encode() + b"\n",
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Line 1"}]}
        }).encode() + b"\n",
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Line 2"}]}
        }).encode() + b"\n",
        json.dumps({
            "type": "result",
            "result": "Line 1\nLine 2",
            "session_id": "test",
            "permission_denials": []
        }).encode() + b"\n",
    ]
    mock_process.stdout = async_iter(events)
    collected = []

    async def callback(line):
        collected.append(line)

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await runner.run("Hello", on_output=callback)

    assert len(collected) == 2
    assert "Line 1" in collected[0]
    assert "Line 2" in collected[1]


@pytest.mark.asyncio
async def test_run_multiline_output(runner, mock_process):
    """Test running Claude with multiline output."""
    mock_process.stdout = async_iter(make_stream_json("First line\nSecond line\nThird line"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await runner.run("Hello")

    assert "First line" in result.text
    assert "Second line" in result.text
    assert "Third line" in result.text


@pytest.mark.asyncio
async def test_compact(runner, mock_process):
    """Test running compaction."""
    mock_process.stdout = async_iter(make_stream_json("Compaction complete"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await runner.compact()

        assert "Compaction complete" in result.text
        call_args = mock_exec.call_args[0]
        assert "--continue" in call_args
        assert "/compact" in call_args


@pytest.mark.asyncio
async def test_cancel_running_process(runner, mock_process):
    """Test cancelling a running process."""
    runner.current_process = mock_process

    result = await runner.cancel()

    assert result is True
    mock_process.terminate.assert_called_once()
    assert runner.current_process is None


@pytest.mark.asyncio
async def test_cancel_no_process(runner):
    """Test cancelling when nothing is running."""
    result = await runner.cancel()
    assert result is False


def test_is_running_true(runner, mock_process):
    """Test is_running when process exists."""
    runner.current_process = mock_process
    assert runner.is_running is True


def test_is_running_false(runner):
    """Test is_running when no process."""
    assert runner.is_running is False


@pytest.mark.asyncio
async def test_run_clears_process_after_completion(runner, mock_process):
    """Test that process reference is cleared after completion."""
    mock_process.stdout = async_iter(make_stream_json("Done"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await runner.run("Hello")

    assert runner.current_process is None


@pytest.mark.asyncio
async def test_run_with_working_directory(mock_process):
    """Test running Claude with custom working directory."""
    mock_process.stdout = async_iter(make_stream_json("Output"))

    runner = ClaudeRunner()
    runner.working_dir = "/custom/path"

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await runner.run("Hello")

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["cwd"].as_posix() == "/custom/path"


@pytest.mark.asyncio
async def test_run_handles_unicode(runner, mock_process):
    """Test handling of unicode output."""
    mock_process.stdout = async_iter(make_stream_json("Hello 世界! 🎉"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await runner.run("Unicode test")

    assert "世界" in result.text
    assert "🎉" in result.text


@pytest.mark.asyncio
async def test_run_with_permission_denials(runner, mock_process):
    """Test running Claude with permission denials."""
    denials = [
        {"tool_name": "Write", "tool_input": {"file_path": "/tmp/test.txt"}, "tool_use_id": "123"}
    ]
    mock_process.stdout = async_iter(make_stream_json("Permission denied", denials))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await runner.run("Write to /tmp/test.txt")

    assert len(result.permission_denials) == 1
    assert result.permission_denials[0].tool_name == "Write"
    assert result.permission_denials[0].tool_input["file_path"] == "/tmp/test.txt"


@pytest.mark.asyncio
async def test_run_with_allowed_tools(runner, mock_process):
    """Test running Claude with allowed tools."""
    mock_process.stdout = async_iter(make_stream_json("Done"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await runner.run("Hello", allowed_tools=["Write:/tmp/*", "Bash:echo *"])

        call_args = mock_exec.call_args[0]
        assert "--allowedTools" in call_args
        idx = call_args.index("--allowedTools")
        assert "Write:/tmp/*,Bash:echo *" in call_args[idx + 1]


@pytest.mark.asyncio
async def test_run_timeout_kills_process(runner, mock_process):
    """Test that run() kills process after timeout."""
    async def never_ending():
        await asyncio.sleep(999)
        return
        yield  # make it an async generator

    mock_process.stdout = never_ending()
    mock_process.returncode = -15

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(TimeoutError):
            await runner.run("Hello", timeout=0.1)

    mock_process.terminate.assert_called()
    assert runner.current_process is None


@pytest.mark.asyncio
async def test_run_timeout_escalates_to_sigkill(runner, mock_process):
    """Test that run() escalates to SIGKILL if SIGTERM doesn't work."""
    async def never_ending():
        await asyncio.sleep(999)
        return
        yield

    mock_process.stdout = never_ending()
    mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_process.kill = MagicMock()
    mock_process.returncode = -9

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(TimeoutError):
            await runner.run("Hello", timeout=0.1)

    mock_process.terminate.assert_called()
    mock_process.kill.assert_called()
    assert runner.current_process is None


@pytest.mark.asyncio
async def test_cancel_escalates_to_sigkill(runner, mock_process):
    """Test that cancel() escalates to SIGKILL if SIGTERM fails."""
    runner.current_process = mock_process
    mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_process.kill = MagicMock()

    result = await runner.cancel()

    assert result is True
    mock_process.terminate.assert_called()
    mock_process.kill.assert_called()
    assert runner.current_process is None


@pytest.mark.asyncio
async def test_run_default_timeout(runner, mock_process):
    """Test that run() uses default 300s timeout and completes normally."""
    mock_process.stdout = async_iter(make_stream_json("OK"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await runner.run("Hello")
        assert result.text == "OK"


# --- SessionManager hierarchical tests ---

from claude_telegram.claude import SessionManager


def test_session_manager_get_session_with_thread():
    sm = SessionManager()
    runner = sm.get_session("/tmp/test", thread_id=42)
    assert runner is not None
    assert runner.working_dir == "/tmp/test"


def test_session_manager_same_session_twice():
    sm = SessionManager()
    r1 = sm.get_session("/tmp/test", thread_id=42)
    r2 = sm.get_session("/tmp/test", thread_id=42)
    assert r1 is r2


def test_session_manager_different_threads():
    sm = SessionManager()
    r1 = sm.get_session("/tmp/test", thread_id=42)
    r2 = sm.get_session("/tmp/test", thread_id=99)
    assert r1 is not r2


def test_session_manager_list_sessions_for_dir():
    sm = SessionManager()
    sm.get_session("/tmp/test", thread_id=42)
    sm.get_session("/tmp/test", thread_id=99)
    sm.get_session("/tmp/other", thread_id=1)
    sessions = sm.list_sessions("/tmp/test")
    assert len(sessions) == 2
    assert 42 in sessions
    assert 99 in sessions


def test_session_manager_list_dirs():
    sm = SessionManager()
    sm.get_session("/tmp/test", thread_id=42)
    sm.get_session("/tmp/test", thread_id=99)
    sm.get_session("/tmp/other", thread_id=1)
    dirs = sm.list_dirs()
    assert len(dirs) == 2
    assert ("/tmp/test", 2) in dirs
    assert ("/tmp/other", 1) in dirs


def test_session_manager_remove_session():
    sm = SessionManager()
    sm.get_session("/tmp/test", thread_id=42)
    sm.get_session("/tmp/test", thread_id=99)
    removed = sm.remove_session("/tmp/test", thread_id=42)
    assert removed is True
    sessions = sm.list_sessions("/tmp/test")
    assert 42 not in sessions
    assert 99 in sessions


def test_session_manager_remove_cleans_empty_dir():
    sm = SessionManager()
    sm.get_session("/tmp/test", thread_id=42)
    sm.remove_session("/tmp/test", thread_id=42)
    assert "/tmp/test" not in sm.sessions


def test_session_manager_any_running():
    sm = SessionManager()
    runner = sm.get_session("/tmp/test", thread_id=42)
    assert sm.any_running() is False
    runner.current_process = MagicMock()
    assert sm.any_running() is True


@pytest.mark.asyncio
async def test_force_kill_refuses_pgid_1():
    """Test that _force_kill refuses to killpg when pgid <= 1 (would kill all user processes)."""
    runner = ClaudeRunner()
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.terminate = MagicMock()
    runner.current_process = mock_proc

    with patch('os.getpgid', return_value=1) as mock_getpgid,          patch('os.killpg') as mock_killpg:
        await runner._force_kill()
        # Should NOT have called killpg (pgid=1 would kill all processes)
        mock_killpg.assert_not_called()
        # Should have fallen back to proc.terminate()
        mock_proc.terminate.assert_called_once()
    assert runner.current_process is None


@pytest.mark.asyncio
async def test_force_kill_normal_pgid():
    """Test that _force_kill works normally with a valid pgid > 1."""
    import signal as sig_module
    runner = ClaudeRunner()
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.terminate = MagicMock()
    runner.current_process = mock_proc

    with patch('os.getpgid', return_value=54321) as mock_getpgid,          patch('os.killpg') as mock_killpg:
        await runner._force_kill()
        # Should have called killpg with the valid pgid
        mock_killpg.assert_called_once_with(54321, sig_module.SIGTERM)
        # Should NOT have called proc.terminate()
        mock_proc.terminate.assert_not_called()
    assert runner.current_process is None


def test_claude_result_error_fields():
    """Test ClaudeResult includes error fields."""
    result = ClaudeResult(text="", error="quota_exceeded", is_quota_error=True)
    assert result.error == "quota_exceeded"
    assert result.is_quota_error is True


def test_claude_result_defaults_no_error():
    """Test ClaudeResult defaults to no error."""
    result = ClaudeResult(text="hello")
    assert result.error is None
    assert result.is_quota_error is False


class AsyncIterator:
    """Helper for async iteration in tests."""
    def __init__(self, items):
        self.items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_execute_detects_error_event():
    """Test that _execute parses error events from stream-json."""
    runner = ClaudeRunner.__new__(ClaudeRunner)
    runner.working_dir = "/tmp/test"
    runner.session_id = None
    runner.last_interaction = None

    error_event = json.dumps({"type": "error", "error": {"message": "Your account has exceeded its quota"}})

    proc = AsyncMock()
    proc.stdout = AsyncIterator([error_event.encode()])
    proc.wait = AsyncMock(return_value=1)
    proc.returncode = 1
    runner.current_process = proc

    result = await runner._execute()
    assert result.error is not None
    assert "exceeded" in result.error.lower()
    assert result.is_quota_error is True


@pytest.mark.asyncio
async def test_execute_detects_nonzero_returncode():
    """Test that _execute flags error on non-zero exit with no result."""
    runner = ClaudeRunner.__new__(ClaudeRunner)
    runner.working_dir = "/tmp/test"
    runner.session_id = None
    runner.last_interaction = None

    proc = AsyncMock()
    proc.stdout = AsyncIterator([])
    proc.wait = AsyncMock(return_value=1)
    proc.returncode = 1
    runner.current_process = proc

    result = await runner._execute()
    assert result.error is not None
    assert "exit" in result.error.lower() or "code" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_no_error_on_success():
    """Test that _execute returns no error on successful run."""
    runner = ClaudeRunner.__new__(ClaudeRunner)
    runner.working_dir = "/tmp/test"
    runner.session_id = None
    runner.last_interaction = None

    result_event = json.dumps({"type": "result", "result": "Hello!", "session_id": "abc123"})

    proc = AsyncMock()
    proc.stdout = AsyncIterator([result_event.encode()])
    proc.wait = AsyncMock(return_value=0)
    proc.returncode = 0
    runner.current_process = proc

    result = await runner._execute()
    assert result.text == "Hello!"
    assert result.error is None
    assert result.is_quota_error is False


@pytest.mark.asyncio
async def test_execute_error_not_flagged_when_result_present():
    """Test that error is not flagged when there is also a result text."""
    runner = ClaudeRunner.__new__(ClaudeRunner)
    runner.working_dir = "/tmp/test"
    runner.session_id = None
    runner.last_interaction = None

    # An error event followed by a result event (non-fatal error)
    error_event = json.dumps({"type": "error", "error": {"message": "rate limit warning"}})
    result_event = json.dumps({"type": "result", "result": "Here is the response", "session_id": "abc"})

    proc = AsyncMock()
    proc.stdout = AsyncIterator([error_event.encode(), result_event.encode()])
    proc.wait = AsyncMock(return_value=0)
    proc.returncode = 0
    runner.current_process = proc

    result = await runner._execute()
    assert result.text == "Here is the response"
    assert result.error is None  # Not flagged because we got a result
    assert result.is_quota_error is False  # returncode=0, so it's just a warning


@pytest.mark.asyncio
async def test_execute_quota_error_in_response_text():
    """Test that quota error is detected when CLI outputs it as assistant text with non-zero exit."""
    runner = ClaudeRunner.__new__(ClaudeRunner)
    runner.working_dir = "/tmp/test"
    runner.session_id = None
    runner.last_interaction = None

    # Claude CLI outputs the quota error as regular assistant text (not as error event)
    assistant_event = json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "You've hit your limit. Resets at 11pm."}]},
    })

    proc = AsyncMock()
    proc.stdout = AsyncIterator([assistant_event.encode()])
    proc.wait = AsyncMock(return_value=1)
    proc.returncode = 1
    runner.current_process = proc

    result = await runner._execute()
    assert result.is_quota_error is True
    # text contains the quota message (used for notification)
    assert "hit your limit" in result.text.lower()
