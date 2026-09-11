"""Tests for the LLM provider chain (Claude -> DeepSeek hot switch)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_telegram import providers
from claude_telegram.claude import ClaudeResult, ClaudeRunner
from claude_telegram.providers import (
    SWITCH_PREFIX,
    FailureKind,
    ProviderState,
    classify_failure,
    provider_env,
    run_with_fallback,
)


@pytest.fixture
def deepseek_key(tmp_path, monkeypatch):
    key = tmp_path / "deepseekapikey"
    key.write_text("  sk-test-123\n")
    monkeypatch.setattr(providers.settings, "deepseek_api_key_file", str(key))
    return key


# --- classification -------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("You've hit your usage limit", FailureKind.QUOTA),
    ("Credit balance is too low", FailureKind.QUOTA),
    ("error: insufficient_quota", FailureKind.QUOTA),
    ("HTTP 429 Too Many Requests", FailureKind.QUOTA),
    ("API Error: 401 invalid x-api-key", FailureKind.AUTH),
    ("403 Forbidden", FailureKind.AUTH),
    ("Invalid API key · Please run /login", FailureKind.AUTH),
    ("API Error: 502 Bad Gateway", FailureKind.TRANSIENT),
    ("read ECONNRESET", FailureKind.TRANSIENT),
    ("something odd happened", FailureKind.UNKNOWN),
])
def test_classify_failure(text, kind):
    assert classify_failure(1, "", text) == kind


def test_classify_ignores_numbers_inside_words():
    assert classify_failure(1, "", "session 14010 ended") == FailureKind.UNKNOWN


# --- state ----------------------------------------------------------------

def test_mark_quota_sets_60min_cooldown(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    assert state.mark("claude", FailureKind.QUOTA, "hit your limit") is True
    assert state.in_cooldown("claude")
    data = json.loads((tmp_path / "s.json").read_text())
    until = datetime.fromisoformat(data["claude"]["unavailable_until"])
    delta = until - datetime.now(timezone.utc)
    assert timedelta(minutes=59) < delta <= timedelta(minutes=60)
    assert data["claude"]["reason"] == "quota"
    # Already quarantined -> not a new entry (no second alert)
    assert state.mark("claude", FailureKind.QUOTA, "again") is False


def test_timeout_and_unknown_do_not_quarantine(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    assert state.mark("claude", FailureKind.TIMEOUT) is False
    assert state.mark("claude", FailureKind.UNKNOWN) is False
    assert not state.in_cooldown("claude")


def test_cooldown_expires(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    state.mark("claude", FailureKind.TRANSIENT)
    later = datetime.now(timezone.utc) + timedelta(minutes=6)
    assert not state.in_cooldown("claude", now=later)


def test_status(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    state.mark("claude", FailureKind.AUTH)
    st = state.status(["claude", "deepseek"])
    assert st["deepseek"] == "ok"
    assert st["claude"].startswith("cooldown until ")


def test_corrupt_state_file_is_ignored(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert not ProviderState(p).in_cooldown("claude")


# --- env ------------------------------------------------------------------

def test_provider_env_claude_is_empty():
    assert provider_env("claude") == {}


def test_provider_env_deepseek_unconfigured():
    assert provider_env("deepseek") is None


def test_provider_env_deepseek(deepseek_key, monkeypatch, tmp_path):
    monkeypatch.setattr(providers.settings, "claude_slim_config_dir", str(tmp_path / "slim"))
    env = provider_env("deepseek")
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test-123"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        assert env[var] == "deepseek-v4-flash"
    assert env["CLAUDE_CODE_ARTIFACT"] == "0"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "slim")
    assert env["ANTHROPIC_API_KEY"] is None  # removed, never inherited


def test_provider_env_deepseek_empty_key(tmp_path, monkeypatch):
    key = tmp_path / "k"
    key.write_text("\n")
    monkeypatch.setattr(providers.settings, "deepseek_api_key_file", str(key))
    assert provider_env("deepseek") is None


# --- ClaudeRunner.run(provider_env=...) -----------------------------------

async def test_runner_merges_provider_env(mock_subprocess, monkeypatch):
    mock_exec, process = mock_subprocess
    process.stdout.__aiter__.return_value = []
    process.returncode = 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-go")
    runner = ClaudeRunner(working_dir="/tmp")
    await runner.run("hi", new_session=True,
                     provider_env={"ANTHROPIC_BASE_URL": "http://x", "ANTHROPIC_API_KEY": None})
    env = mock_exec.call_args.kwargs["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://x"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_TELEGRAM_BOT"] == "1"


# --- run_with_fallback ----------------------------------------------------

def _runner(*results):
    r = MagicMock()
    r.run = AsyncMock(side_effect=list(results))
    r.session_id = None
    r.working_dir = "/tmp/x"
    r.short_name = "x"
    return r


def _quota():
    return ClaudeResult(text="", error="You've hit your limit", is_quota_error=True,
                        failure_kind="quota")


def _ok(text="hello", session_id="s1"):
    return ClaudeResult(text=text, session_id=session_id)


async def test_success_single_call(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    runner = _runner(_ok())
    notify = AsyncMock()
    res = await run_with_fallback(runner, "hi", state=state, notify=notify, new_session=True)
    assert res.text == "hello"
    assert res.provider == "claude"
    assert runner.run.call_count == 1
    assert runner.run.call_args.kwargs["provider_env"] == {}
    notify.assert_not_called()


async def test_quota_switches_to_deepseek(tmp_path, deepseek_key):
    state = ProviderState(tmp_path / "s.json")
    runner = _runner(_quota(), _ok("from deepseek"))
    notify = AsyncMock()
    res = await run_with_fallback(runner, "hi", state=state, notify=notify,
                                  model="opus", new_session=True)
    assert runner.run.call_count == 2
    second = runner.run.call_args_list[1].kwargs
    assert second["provider_env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-test-123"
    assert second["model"] is None  # ANTHROPIC_MODEL decides
    assert res.text.startswith(SWITCH_PREFIX)
    assert "from deepseek" in res.text
    assert res.provider == "deepseek"
    assert state.in_cooldown("claude")
    notify.assert_awaited_once()


async def test_claude_in_cooldown_is_not_spawned(tmp_path, deepseek_key):
    state = ProviderState(tmp_path / "s.json")
    state.mark("claude", FailureKind.QUOTA)
    runner = _runner(_ok("ds"))
    notify = AsyncMock()
    res = await run_with_fallback(runner, "hi", state=state, notify=notify, new_session=True)
    assert runner.run.call_count == 1
    assert runner.run.call_args.kwargs["provider_env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-test-123"
    assert res.text.startswith(SWITCH_PREFIX)
    notify.assert_not_called()  # already quarantined: no new alert


async def test_no_deepseek_returns_quota_result(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    runner = _runner(_quota())
    notify = AsyncMock()
    res = await run_with_fallback(runner, "hi", state=state, notify=notify, new_session=True)
    assert res.is_quota_error
    assert runner.run.call_count == 1
    # Nothing left in the chain -> urgent
    assert notify.await_args.args[1] == "urgent"


async def test_all_in_cooldown_returns_quota_error_without_spawn(tmp_path):
    state = ProviderState(tmp_path / "s.json")
    state.mark("claude", FailureKind.QUOTA)
    runner = _runner()
    res = await run_with_fallback(runner, "hi", state=state, notify=AsyncMock(), new_session=True)
    assert res.is_quota_error
    runner.run.assert_not_called()


async def test_transient_failure_does_not_switch(tmp_path, deepseek_key):
    state = ProviderState(tmp_path / "s.json")
    err = ClaudeResult(text="", error="API Error: 502", failure_kind="transient")
    runner = _runner(err)
    res = await run_with_fallback(runner, "hi", state=state, notify=AsyncMock(), new_session=True)
    assert res is err
    assert runner.run.call_count == 1


async def test_resume_failure_falls_back_to_summary(tmp_path, deepseek_key):
    state = ProviderState(tmp_path / "s.json")
    runner = _runner(_quota(), ClaudeResult(text="", error="No conversation found"), _ok("fresh"))
    runner.session_id = "sess-1"
    with patch("claude_telegram.providers.read_session_messages",
               return_value=[{"role": "user", "text": "bonjour ici"},
                             {"role": "assistant", "text": "salut"}]):
        res = await run_with_fallback(runner, "et maintenant ?", state=state,
                                      notify=AsyncMock(), continue_session=True)
    assert runner.run.call_count == 3
    third_msg = runner.run.call_args_list[2].args[0]
    assert "bonjour ici" in third_msg and "et maintenant ?" in third_msg
    assert res.text.startswith(SWITCH_PREFIX)


async def test_deepseek_failure_is_returned_without_second_switch(tmp_path, deepseek_key):
    state = ProviderState(tmp_path / "s.json")
    ds_err = ClaudeResult(text="", error="401 invalid key", failure_kind="auth")
    runner = _runner(_quota(), ds_err)
    res = await run_with_fallback(runner, "hi", state=state, notify=AsyncMock(), new_session=True)
    assert runner.run.call_count == 2
    assert res.error
    assert state.in_cooldown("deepseek")
