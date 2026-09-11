"""LLM provider chain for interactive sessions: Claude -> DeepSeek.

DeepSeek is not a new SDK: it is the same Claude CLI pointed at DeepSeek's
Anthropic-compatible endpoint through environment variables.

A provider that fails on quota/auth is quarantined in a JSON state file
shared with the pipelines (LLM_PROVIDER_STATE_PATH), so it is skipped without
spawning a process until its window resets. One alert per quarantine entry.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import shutil
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from .claude import (
    CLAUDE_DIR,
    ClaudeResult,
    _dir_to_claude_name,
    find_latest_session,
    read_session_messages,
)
from .config import settings

logger = logging.getLogger(__name__)

SWITCH_PREFIX = "⚠️ Claude indisponible (limite) → réponse DeepSeek."

Notifier = Callable[[str, str], Awaitable[None]]


class FailureKind(str, Enum):
    QUOTA = "quota"
    AUTH = "auth"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNKNOWN = "unknown"


COOLDOWN_S: dict[FailureKind, int] = {
    FailureKind.QUOTA: 60 * 60,
    FailureKind.AUTH: 24 * 60 * 60,
    FailureKind.TRANSIENT: 5 * 60,
    FailureKind.TIMEOUT: 0,
    FailureKind.UNSUPPORTED_MODEL: 0,
    FailureKind.UNKNOWN: 0,
}

_QUOTA_KEYWORDS = (
    "quota", "billing", "rate_limit", "rate limit", "overloaded",
    "credit balance", "quota exceeded", "spending limit", "hit your limit",
    "usage limit", "insufficient_quota",
)
_QUOTA_RE = re.compile(r"(?<!\d)429(?!\d)")
_AUTH_KEYWORDS = (
    "unauthorized", "invalid api key", "invalid x-api-key", "authentication_error",
    "authentication failed", "please run /login", "oauth token has expired",
)
_AUTH_RE = re.compile(r"(?<!\d)40[13](?!\d)")
_TRANSIENT_KEYWORDS = (
    "econnreset", "econnrefused", "etimedout", "enotfound", "socket hang up",
    "connection error", "connection reset", "network error", "fetch failed",
    "internal server error", "bad gateway", "service unavailable", "gateway timeout",
)
_TRANSIENT_RE = re.compile(r"(?<!\d)5\d\d(?!\d)")


def classify_failure(returncode: int | None, stdout: str = "", stderr: str = "") -> FailureKind:
    """Classify a failed CLI run. Order matters: quota, then auth, then transient."""
    text = f"{stdout or ''}\n{stderr or ''}".lower()
    if any(kw in text for kw in _QUOTA_KEYWORDS) or _QUOTA_RE.search(text):
        return FailureKind.QUOTA
    if any(kw in text for kw in _AUTH_KEYWORDS) or _AUTH_RE.search(text):
        return FailureKind.AUTH
    if any(kw in text for kw in _TRANSIENT_KEYWORDS) or _TRANSIENT_RE.search(text):
        return FailureKind.TRANSIENT
    return FailureKind.UNKNOWN


# --- shared cooldown state --------------------------------------------------

def state_path() -> Path:
    raw = settings.llm_provider_state_path or "data/llm-provider-state.json"
    return Path(raw).expanduser()


def _parse_ts(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class ProviderState:
    """`{provider: {unavailable_until, reason, last_error, since}}`, fcntl-locked."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _lock(self, exclusive: bool):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a")
        fcntl.flock(fh, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        return fh

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _until(self, entry) -> datetime | None:
        if not isinstance(entry, dict):
            return None
        return _parse_ts(entry.get("unavailable_until"))

    def in_cooldown(self, provider: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        try:
            with self._lock(exclusive=False):
                entry = self._load().get(provider)
        except OSError:
            return False
        until = self._until(entry)
        return bool(until and until > now)

    def mark(self, provider: str, kind: FailureKind, error: str = "") -> bool:
        """Quarantine `provider` for its failure kind. True only on a new entry."""
        cooldown = COOLDOWN_S.get(FailureKind(kind), 0)
        if cooldown <= 0:
            return False
        now = datetime.now(timezone.utc)
        with self._lock(exclusive=True):
            data = self._load()
            previous = data.get(provider)
            prev_until = self._until(previous)
            already = bool(prev_until and prev_until > now)
            data[provider] = {
                "unavailable_until": (now + timedelta(seconds=cooldown)).isoformat(timespec="seconds"),
                "reason": FailureKind(kind).value,
                "last_error": (error or "")[:500],
                "since": previous.get("since") if already else now.isoformat(timespec="seconds"),
            }
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        return not already

    def clear(self, provider: str) -> None:
        with self._lock(exclusive=True):
            data = self._load()
            if data.pop(provider, None) is not None:
                self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def until(self, provider: str) -> datetime | None:
        try:
            with self._lock(exclusive=False):
                return self._until(self._load().get(provider))
        except OSError:
            return None

    def status(self, providers: list[str]) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        with self._lock(exclusive=False):
            data = self._load()
        out = {}
        for name in providers:
            until = self._until(data.get(name))
            if until and until > now:
                out[name] = f"cooldown until {until.astimezone().strftime('%Y-%m-%d %H:%M')} ({data[name].get('reason')})"
            else:
                out[name] = "ok"
        return out


# --- per-provider environment -------------------------------------------------

def provider_env(name: str) -> dict[str, str | None] | None:
    """Env overrides for a provider (None value = remove the variable).

    Returns None when the provider is unknown or not configured.
    """
    if name == "claude":
        return {}
    if name != "deepseek" or not settings.deepseek_api_key_file:
        return None
    try:
        token = Path(settings.deepseek_api_key_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not token:
        return None
    model = settings.deepseek_model
    env: dict[str, str | None] = {
        "ANTHROPIC_BASE_URL": settings.deepseek_base_url,
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": None,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "API_TIMEOUT_MS": "3000000",
        "CLAUDE_CODE_ARTIFACT": "0",
    }
    if settings.claude_slim_config_dir:
        env["CLAUDE_CONFIG_DIR"] = str(Path(settings.claude_slim_config_dir).expanduser())
    return env


def interactive_chain() -> list[str]:
    return [p.strip() for p in settings.llm_interactive_chain.split(",") if p.strip()]


def telegram_notifier(bot) -> Notifier:
    """Alert callback posting to the bot's own chat (plain text)."""
    async def _notify(text: str, severity: str) -> None:
        from .adapters.telegram.api import send_message
        prefix = "🚨 " if severity == "urgent" else ""
        await send_message(prefix + text, chat_id=bot.chat_id, parse_mode=None, api_url=bot.api_url)
    return _notify


# --- hot switch ---------------------------------------------------------------

def _failure_kind(result: ClaudeResult) -> FailureKind | None:
    """None when the run succeeded."""
    if not (result.error or result.is_quota_error):
        return None
    if result.failure_kind:
        try:
            return FailureKind(result.failure_kind)
        except ValueError:
            pass
    kind = classify_failure(1, result.text or "", result.error or "")
    if kind == FailureKind.UNKNOWN and result.is_quota_error:
        kind = FailureKind.QUOTA
    return kind


def _copy_transcript(session_id: str, working_dir: str | None, src_root: Path, dst_root: Path) -> None:
    if not (session_id and working_dir) or src_root == dst_root:
        return
    try:
        name = _dir_to_claude_name(working_dir)
        src = src_root / "projects" / name / f"{session_id}.jsonl"
        dst = dst_root / "projects" / name / f"{session_id}.jsonl"
        if not src.exists():
            return
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except OSError as e:
        logger.warning(f"Transcript copy {session_id} {src_root} -> {dst_root} failed: {e}")


def _summary_prompt(session_id: str, working_dir: str | None, message: str) -> str:
    messages = []
    if session_id and working_dir:
        messages = read_session_messages(session_id, working_dir, last_n=20) or []
    if not messages:
        return message
    lines = []
    for m in messages:
        who = "Thomas" if m.get("role") == "user" else "Assistant"
        lines.append(f"- {who} : {m.get('text', '')[:800]}")
    return (
        "[Reprise après bascule de fournisseur LLM — la session précédente n'a pas pu être reprise.\n"
        "Derniers échanges :\n" + "\n".join(lines) + "]\n\n" + message
    )


async def _run_provider(runner, name: str, env: dict, message: str, is_fallback: bool, run_kwargs: dict) -> ClaudeResult:
    kwargs = dict(run_kwargs)
    kwargs["provider_env"] = env
    if name != "claude":
        kwargs["model"] = None  # ANTHROPIC_MODEL / DEFAULT_*_MODEL decide
    if not is_fallback:
        return await runner.run(message, **kwargs)

    working_dir = getattr(runner, "working_dir", None)
    working_dir = working_dir if isinstance(working_dir, str) else None
    alt_root = Path(env["CLAUDE_CONFIG_DIR"]) if env.get("CLAUDE_CONFIG_DIR") else CLAUDE_DIR

    session_id = None
    if not kwargs.get("new_session"):
        session_id = runner.session_id if isinstance(getattr(runner, "session_id", None), str) else None
        if not session_id and kwargs.get("continue_session") and working_dir:
            session_id = find_latest_session(working_dir)
            if session_id:
                runner.session_id = session_id
    if not session_id:
        return await runner.run(message, **kwargs)

    # 1) Resume the same session under the fallback provider
    _copy_transcript(session_id, working_dir, CLAUDE_DIR, alt_root)
    result = await runner.run(message, **kwargs)
    kind = _failure_kind(result)
    if kind in (None, FailureKind.QUOTA, FailureKind.AUTH):
        if kind is None:
            _copy_transcript(result.session_id or session_id, working_dir, alt_root, CLAUDE_DIR)
        return result

    # 2) Resume failed: fresh session seeded with the last exchanges
    logger.warning(f"{name}: resume of {session_id} failed ({result.error!r}), starting fresh with summary")
    runner.session_id = None
    kwargs["new_session"] = False
    kwargs["continue_session"] = False
    result = await runner.run(_summary_prompt(session_id, working_dir, message), **kwargs)
    if _failure_kind(result) is None and result.session_id:
        _copy_transcript(result.session_id, working_dir, alt_root, CLAUDE_DIR)
    return result


async def _alert(notify: Notifier | None, text: str, severity: str) -> None:
    logger.warning(f"[llm-provider/{severity}] {text}")
    if not notify:
        return
    try:
        await notify(text, severity)
    except Exception as e:
        logger.error(f"LLM provider alert failed: {e}")


async def run_with_fallback(
    runner,
    message: str,
    *,
    notify: Notifier | None = None,
    state: ProviderState | None = None,
    chain: list[str] | None = None,
    **run_kwargs,
) -> ClaudeResult:
    """Run `runner.run(message, **run_kwargs)` through the interactive chain.

    Switches provider only on quota/auth, at most once per request. Other
    failures (transient, unknown) are returned as-is; TimeoutError propagates.
    """
    state = state or ProviderState(state_path())
    chain = chain or interactive_chain()
    primary = chain[0] if chain else "claude"

    def available(name: str) -> bool:
        return provider_env(name) is not None and not state.in_cooldown(name)

    last: ClaudeResult | None = None
    attempts = 0
    for name in chain:
        env = provider_env(name)
        if env is None or state.in_cooldown(name):
            continue
        if attempts >= 2:
            break
        is_fallback = name != primary
        result = await _run_provider(runner, name, env, message, is_fallback, run_kwargs)
        attempts += 1
        result.provider = name

        kind = _failure_kind(result)
        if kind is None:
            if is_fallback:
                result.text = f"{SWITCH_PREFIX}\n\n{result.text}"
            return result

        result.failure_kind = kind.value
        if kind not in (FailureKind.QUOTA, FailureKind.AUTH):
            return result

        if state.mark(name, kind, result.error or result.text or ""):
            until = state.until(name)
            until_txt = until.astimezone().strftime("%H:%M") if until else "?"
            remaining = [n for n in chain if n != name and available(n)]
            nxt = f"bascule sur {remaining[0]}" if remaining else "plus aucun fournisseur LLM disponible"
            await _alert(
                notify,
                f"LLM {name} en quarantaine ({kind.value}) jusqu'à {until_txt} — {nxt}.",
                "normal" if remaining else "urgent",
            )
        last = result

    if last is None:
        return ClaudeResult(
            text="",
            error="Aucun fournisseur LLM disponible (tous en quarantaine ou non configurés).",
            is_quota_error=True,
            failure_kind=FailureKind.QUOTA.value,
            provider=primary,
        )
    last.is_quota_error = True
    return last
