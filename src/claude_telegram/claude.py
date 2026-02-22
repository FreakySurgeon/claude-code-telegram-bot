"""Claude Code runner - spawns and manages Claude processes."""

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings


@dataclass
class PermissionDenial:
    """A permission that was denied during Claude execution."""
    tool_name: str
    tool_input: dict
    tool_use_id: str = ""


@dataclass
class ClaudeResult:
    """Result from running Claude, including any permission denials."""
    text: str
    permission_denials: list[PermissionDenial] = field(default_factory=list)
    session_id: str | None = None
    error: str | None = None
    is_quota_error: bool = False

logger = logging.getLogger(__name__)

CONVERSATION_TIMEOUT = timedelta(minutes=10)
CLAUDE_DIR = Path.home() / ".claude"


def get_project_dir(working_dir: str) -> Path | None:
    """Get the Claude project directory for a working directory."""
    # Claude stores projects in ~/.claude/projects/<path-with-dashes>/
    # e.g., /Users/foo/bar -> -Users-foo-bar
    projects_dir = CLAUDE_DIR / "projects"
    if not projects_dir.exists():
        return None

    # Convert path to Claude's format: /Users/foo/bar -> -Users-foo-bar
    # Claude replaces / . and _ with dashes
    abs_path = str(Path(working_dir).resolve())
    claude_dir_name = abs_path.replace("/", "-").replace(".", "-").replace("_", "-")

    # Check for exact path match
    project_path = projects_dir / claude_dir_name
    if project_path.exists() and project_path.is_dir():
        return project_path

    # Fallback: look for any project dir that might match
    dir_name = working_dir.split("/")[-1].replace(".", "-").replace("_", "-")
    for project_path in projects_dir.iterdir():
        if project_path.is_dir() and project_path.name.endswith(f"-{dir_name}"):
            return project_path

    return None


def find_latest_session(working_dir: str) -> str | None:
    """Find the most recent session ID for a working directory."""
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return None

    # Find most recent .jsonl file (excluding agent-* files)
    sessions = [
        f for f in project_dir.glob("*.jsonl")
        if not f.name.startswith("agent-")
    ]

    if not sessions:
        return None

    # Get most recently modified
    latest = max(sessions, key=lambda f: f.stat().st_mtime)
    # Return session ID (filename without .jsonl)
    return latest.stem


def delete_session(session_id: str, working_dir: str) -> bool:
    """Delete a session .jsonl file from disk. Returns True if deleted."""
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return False
    session_file = project_dir / f"{session_id}.jsonl"
    if session_file.exists():
        session_file.unlink()
        logger.info(f"Deleted session file: {session_file.name}")
        return True
    return False


def _dir_to_claude_name(path: str) -> str:
    """Convert a filesystem path to Claude's project directory name format."""
    return str(Path(path).resolve()).replace("/", "-").replace(".", "-").replace("_", "-")


def find_session_working_dir(session_id: str) -> str | None:
    """Find the working directory for a session by scanning all project directories.

    Returns the reconstructed working_dir (e.g. /home/thomas/projects/personal-org)
    or None if not found.
    """
    projects_dir = CLAUDE_DIR / "projects"
    if not projects_dir.exists():
        return None

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        session_file = project_dir / f"{session_id}.jsonl"
        if session_file.exists():
            # Reconstruct working_dir from project dir name
            # The name has dashes replacing / . and _, so it's ambiguous to reverse.
            # Strategy: try naive replacement, then check common parent dirs.
            dir_name = project_dir.name.lstrip("-")
            candidate = "/" + dir_name.replace("-", "/")
            if Path(candidate).is_dir():
                return candidate
            # Try parent dirs that exist and check children with underscores/dots
            # e.g. -home-thomas-media-server-docker -> /home/thomas/media_server_docker
            parts = dir_name.split("-")
            for i in range(len(parts) - 1, 0, -1):
                parent = "/" + "/".join(parts[:i])
                if Path(parent).is_dir():
                    # Check children — try combining remaining parts with _ and .
                    remaining = "-".join(parts[i:])
                    for child in Path(parent).iterdir():
                        if child.is_dir() and _dir_to_claude_name(str(child)).lstrip("-") == dir_name:
                            return str(child)
                    break
            # Last resort: return the naive candidate
            return candidate
    return None


def list_recent_sessions(working_dir: str, limit: int = 8) -> list[dict]:
    """List recent sessions for a working directory.

    Returns list of {"id": str, "timestamp": str, "first_message": str} dicts,
    sorted by most recent first.
    """
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return []

    session_files = sorted(
        (f for f in project_dir.glob("*.jsonl")
         if not f.name.startswith("agent-") and f.stat().st_size > 0),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:limit]

    results = []
    for sf in session_files:
        timestamp = None
        first_message = None
        try:
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "queue-operation" and not timestamp:
                            timestamp = data.get("timestamp", "")
                        if data.get("type") == "user" and not first_message:
                            content = data.get("message", {}).get("content", [])
                            if isinstance(content, str):
                                first_message = content.strip()
                            elif isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        first_message = c["text"].strip()
                                        break
                        if timestamp and first_message:
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

        if first_message:
            results.append({
                "id": sf.stem,
                "timestamp": timestamp or "",
                "first_message": first_message,
            })

    return results


def read_session_messages(session_id: str, working_dir: str, last_n: int = 5) -> list[dict] | None:
    """Read last N user/assistant messages from a session file.

    Returns list of {"role": "user"|"assistant", "text": str} or None if not found.
    """
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return None

    session_file = project_dir / f"{session_id}.jsonl"
    if not session_file.exists() or session_file.stat().st_size == 0:
        return None

    messages = []
    CONTINUATION_MARKER = "continued from a previous conversation"
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "user":
                        content = data.get("message", {}).get("content", [])
                        text = ""
                        if isinstance(content, str):
                            text = content.strip()
                        elif isinstance(content, list):
                            parts = []
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    parts.append(c["text"].strip())
                            text = "\n".join(parts)
                        # Context compaction boundary — reset to only keep current segment
                        if CONTINUATION_MARKER in text.lower()[:200]:
                            messages.clear()
                            continue
                        if text and len(text) > 10 and not text.startswith("[Request"):
                            messages.append({"role": "user", "text": text})
                    elif data.get("type") == "assistant":
                        content = data.get("message", {}).get("content", [])
                        parts = []
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    parts.append(c["text"].strip())
                        text = "\n".join(parts)
                        if text:
                            messages.append({"role": "assistant", "text": text})
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Failed to read session file {session_id}: {e}")
        return None

    return messages[-last_n:] if messages else []


def get_session_permission_mode(working_dir: str) -> str | None:
    """Check if a session was started with bypass permissions mode."""
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return None

    sessions = [
        f for f in project_dir.glob("*.jsonl")
        if not f.name.startswith("agent-")
    ]

    if not sessions:
        return None

    latest = max(sessions, key=lambda f: f.stat().st_mtime)

    # Read first few lines to find permissionMode
    try:
        with open(latest, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 10:  # Only check first 10 lines
                    break
                try:
                    data = json.loads(line)
                    if "permissionMode" in data:
                        return data["permissionMode"]
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return None


class ClaudeRunner:
    """Runs Claude Code for a specific working directory."""

    def __init__(self, working_dir: str | None = None):
        self.cli_path = settings.claude_cli_path
        self.working_dir = working_dir or settings.claude_working_dir
        self.current_process: asyncio.subprocess.Process | None = None
        self.last_interaction: datetime | None = None
        self.session_id: str | None = None  # Track session ID for --resume
        self.context_shown: bool = False  # Track if we've shown context for resumed session

    def get_session_context(self) -> str | None:
        """Get the last few user messages from a stored session."""
        if not self.working_dir:
            return None

        project_dir = get_project_dir(self.working_dir)
        if not project_dir:
            return None

        # Find most recent session file
        sessions = [
            f for f in project_dir.glob("*.jsonl")
            if not f.name.startswith("agent-") and f.stat().st_size > 0
        ]
        if not sessions:
            return None

        latest = max(sessions, key=lambda f: f.stat().st_mtime)

        # Read and parse user messages (read-only — do NOT set self.session_id here,
        # session resumption is handled exclusively by run())
        messages = []
        try:
            with open(latest, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "user":
                            content = data.get("message", {}).get("content", [])
                            # Content can be a string or a list
                            if isinstance(content, str):
                                text = content.strip()
                                if text and len(text) > 10 and not text.startswith("[Request"):
                                    messages.append(text[:120])
                            elif isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        text = c.get("text", "").strip()
                                        if text and len(text) > 10 and not text.startswith("[Request"):
                                            messages.append(text[:120])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read session file: {e}")
            return None

        if not messages:
            return None

        self.context_shown = True
        # Return last 5 messages as bullet points
        return "\n".join(f"• {m}" for m in messages[-5:])

    async def run(
        self,
        message: str,
        *,
        continue_session: bool = False,
        new_session: bool = False,
        on_output: callable = None,
        allowed_tools: list[str] | None = None,
        bypass_permissions: bool = False,
        system_prompt: str | None = None,
        mcp_config: str | None = None,
        timeout: float = 300,
    ) -> ClaudeResult:
        """
        Run Claude Code with a message.

        Args:
            message: The prompt to send to Claude
            continue_session: If True, resume the session
            new_session: If True, force a new session (ignore stored session_id)
            on_output: Optional callback for streaming output
            allowed_tools: Optional list of tools to allow (e.g., ["Write", "Bash(echo:*)"])
            bypass_permissions: If True, skip all permission prompts
            system_prompt: Optional system prompt to append (e.g., GTD bot)
            mcp_config: Optional path to MCP config file
            timeout: Timeout in seconds (default 300s / 5 minutes)

        Returns:
            ClaudeResult with response text and any permission denials
        """
        cmd = [self.cli_path, "--print", "--output-format", "stream-json", "--verbose"]

        # Add bypass permissions if specified
        if bypass_permissions:
            cmd.append("--dangerously-skip-permissions")

        # Add allowed tools if specified
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        # Add system prompt if specified (GTD bot)
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Add MCP config if specified
        if mcp_config:
            cmd.extend(["--mcp-config", mcp_config])

        # Session handling: new_session forces a fresh start, otherwise try to resume
        if not new_session:
            if self.session_id:
                # We already have a session ID from previous run
                cmd.extend(["--resume", self.session_id])
            elif continue_session:
                # Only search disk for sessions when explicitly continuing
                if self.working_dir:
                    session_id = find_latest_session(self.working_dir)
                    if session_id:
                        cmd.extend(["--resume", session_id])
                        self.session_id = session_id
                        logger.info(f"Resuming stored session {session_id} for {self.short_name}")
                    else:
                        cmd.append("--continue")
                else:
                    cmd.append("--continue")

        # Prompt is a positional argument, not a flag
        cmd.append(message)

        logger.info(f"Running: {' '.join(cmd)} in {self.working_dir or 'cwd'}")

        cwd = Path(self.working_dir) if self.working_dir else None

        # Set environment variable to prevent hook from sending duplicate notifications
        env = os.environ.copy()
        env["CLAUDE_TELEGRAM_BOT"] = "1"

        self.current_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=1024 * 1024,  # 1 MiB line buffer (default 64 KiB too small for large Claude JSON events)
            cwd=cwd,
            env=env,
            start_new_session=True,  # Own process group so we can kill MCP children too
        )

        try:
            return await asyncio.wait_for(
                self._execute(on_output=on_output, new_session=new_session),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            await self._force_kill()
            raise TimeoutError(f"Claude process timed out after {timeout}s")

    async def compact(self) -> ClaudeResult:
        """Run compaction on the current session."""
        return await self.run("/compact", continue_session=True)

    async def _execute(
        self,
        *,
        on_output: callable = None,
        new_session: bool = False,
    ) -> ClaudeResult:
        """Internal: read stdout, wait for process, and return result."""
        # Parse stream-json output
        result_text = ""
        accumulated_text = ""
        permission_denials = []
        result_session_id = None
        error_message = None

        async for line in self.current_process.stdout:
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue

            try:
                event = json.loads(decoded)
                event_type = event.get("type")

                # Extract result text from the final result event
                if event_type == "result":
                    result_text = event.get("result", "")
                    result_session_id = event.get("session_id")
                    # Parse permission denials
                    for denial in event.get("permission_denials", []):
                        permission_denials.append(PermissionDenial(
                            tool_name=denial.get("tool_name", ""),
                            tool_input=denial.get("tool_input", {}),
                            tool_use_id=denial.get("tool_use_id", ""),
                        ))

                # Stream assistant text content for real-time output
                if event_type == "assistant":
                    content = event.get("message", {}).get("content", [])
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c.get("text", "")
                            accumulated_text += text
                            if on_output:
                                await on_output(text)

                # Capture error events
                if event_type == "error":
                    err = event.get("error", {})
                    if isinstance(err, dict):
                        error_message = err.get("message", str(err))
                    else:
                        error_message = str(event.get("message", event.get("error", "unknown error")))
                    logger.error(f"Claude error event: {error_message}")

            except json.JSONDecodeError:
                # Capture meaningful non-JSON stderr output as potential error
                if decoded and not error_message:
                    error_message = decoded
                logger.debug(f"Non-JSON output: {decoded}")
                continue

        proc = self.current_process
        if proc:
            await proc.wait()
        self.current_process = None
        self.last_interaction = datetime.now()

        returncode = proc.returncode if proc else None
        if returncode and returncode != 0 and not error_message:
            error_message = f"Claude process exited with code {returncode}"
            logger.warning(error_message)

        # Update session ID after run (but not for one-off new_session runs like email/cron)
        if not new_session:
            if result_session_id:
                self.session_id = result_session_id
            elif self.working_dir:
                new_session_id = find_latest_session(self.working_dir)
                if new_session_id:
                    self.session_id = new_session_id

        run_session_id = result_session_id or self.session_id
        if run_session_id:
            logger.info(f"Session ID for {self.short_name}: {run_session_id}")

        # Detect quota-related errors
        quota_keywords = (
            "quota", "billing", "rate_limit", "rate limit",
            "overloaded", "credit balance", "quota exceeded",
            "spending limit", "hit your limit", "usage limit",
        )
        is_quota = False
        if error_message:
            lower = error_message.lower()
            is_quota = any(kw in lower for kw in quota_keywords)

        # Claude CLI sometimes emits quota errors as regular assistant text
        # (not as error events), so also check the response text when
        # the process exited with a non-zero code.
        response_text = result_text or accumulated_text
        if not is_quota and returncode and returncode != 0 and response_text:
            lower_resp = response_text.lower()
            is_quota = any(kw in lower_resp for kw in quota_keywords)
            if is_quota:
                error_message = response_text
                logger.warning(f"Quota error detected in response text: {response_text[:100]}")

        has_result = bool(response_text)
        # is_quota_error only when the request actually failed:
        # either no result, or process exited with error code
        failed = not has_result or (returncode is not None and returncode != 0)
        return ClaudeResult(
            text=response_text,
            permission_denials=permission_denials,
            session_id=run_session_id,
            error=error_message if not has_result else None,
            is_quota_error=is_quota and failed,
        )

    async def _force_kill(self):
        """Kill the current process and its children (MCP servers) with SIGTERM -> SIGKILL."""
        if not self.current_process:
            return
        proc = self.current_process
        self.current_process = None
        # Kill entire process group (Claude + MCP server children)
        # SAFETY: pgid <= 1 would kill all user processes (kill(-1, SIGTERM))
        try:
            pgid = os.getpgid(proc.pid)
            if pgid > 1:
                os.killpg(pgid, signal.SIGTERM)
            else:
                logger.warning("Refusing to killpg(%s, SIGTERM) — would kill all user processes", pgid)
                proc.terminate()
        except (ProcessLookupError, OSError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                if pgid > 1:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    logger.warning("Refusing to killpg(%s, SIGKILL) — would kill all user processes", pgid)
                    proc.kill()
            except (ProcessLookupError, OSError):
                proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass

    async def cancel(self) -> bool:
        """Cancel the currently running Claude process."""
        if self.current_process:
            await self._force_kill()
            return True
        return False

    @property
    def is_running(self) -> bool:
        """Check if Claude is currently running."""
        return self.current_process is not None

    def is_in_conversation(self) -> bool:
        """Check if we're in an active conversation (should auto-continue)."""
        if self.last_interaction is None:
            return False
        return datetime.now() - self.last_interaction < CONVERSATION_TIMEOUT

    @property
    def short_name(self) -> str:
        """Get a short display name for this session."""
        if not self.working_dir:
            return "default"
        return Path(self.working_dir).name


class SessionManager:
    """Manages multiple Claude sessions across directories and topics."""

    def __init__(self):
        self.sessions: dict[str, dict[int, ClaudeRunner]] = {}
        default_dir = settings.claude_working_dir or str(Path.home())
        self.default_dir: str = default_dir

    def get_session(self, working_dir: str | None = None, *, thread_id: int = 0) -> ClaudeRunner:
        """Get or create a session for the given directory + thread."""
        dir_key = working_dir or self.default_dir
        if dir_key not in self.sessions:
            self.sessions[dir_key] = {}
        if thread_id not in self.sessions[dir_key]:
            self.sessions[dir_key][thread_id] = ClaudeRunner(working_dir=dir_key)
            logger.info(f"Created new session for: {dir_key} thread={thread_id}")
        return self.sessions[dir_key][thread_id]

    def find_by_thread(self, thread_id: int) -> ClaudeRunner | None:
        """Find an existing session by thread_id across all working dirs."""
        for threads in self.sessions.values():
            if thread_id in threads:
                return threads[thread_id]
        return None

    def list_sessions(self, working_dir: str | None = None) -> dict[int, ClaudeRunner] | list[tuple[str, ClaudeRunner]]:
        """List sessions. With working_dir: return threads dict. Without: return legacy list."""
        if working_dir is not None:
            return dict(self.sessions.get(working_dir, {}))
        # Legacy: return flat list of (dir, runner) for thread_id=0
        result = []
        for d, threads in self.sessions.items():
            if 0 in threads:
                result.append((d, threads[0]))
            else:
                # Pick first thread if no thread 0
                first = next(iter(threads.values()))
                result.append((d, first))
        return result

    def list_dirs(self) -> list[tuple[str, int]]:
        """List all directories with their session count."""
        return [(d, len(threads)) for d, threads in self.sessions.items()]

    def remove_session(self, working_dir: str, *, thread_id: int | None = None) -> bool:
        """Remove a specific session. If thread_id is None, use legacy behavior."""
        if thread_id is not None:
            # New hierarchical behavior
            if working_dir not in self.sessions:
                return False
            threads = self.sessions[working_dir]
            if thread_id not in threads:
                return False
            if threads[thread_id].is_running:
                return False
            del threads[thread_id]
            if not threads:
                del self.sessions[working_dir]
            return True
        else:
            # Legacy behavior (used by /rmdir in main.py)
            if not working_dir.startswith("/") and not working_dir.startswith("~"):
                working_dir = f"~/{working_dir}"
            resolved = str(Path(working_dir).expanduser().resolve())

            if resolved in self.sessions:
                threads = self.sessions[resolved]
                # Don't remove if any thread is running
                if any(r.is_running for r in threads.values()):
                    return False
                del self.sessions[resolved]
                # If we removed the current dir, switch to another or default
                if self.default_dir == resolved:
                    if self.sessions:
                        self.default_dir = next(iter(self.sessions.keys()))
                    else:
                        self.default_dir = str(Path.home())
                return True
            return False

    def any_running(self) -> bool:
        return any(
            runner.is_running
            for threads in self.sessions.values()
            for runner in threads.values()
        )

    def get_running_session(self) -> ClaudeRunner | None:
        for threads in self.sessions.values():
            for runner in threads.values():
                if runner.is_running:
                    return runner
        return None

    # Temporary backward-compat (removed in Task 5)
    @property
    def current_dir(self) -> str:
        return self.default_dir

    def get_current_session(self) -> ClaudeRunner:
        return self.get_session(self.default_dir, thread_id=0)

    def switch_session(self, working_dir: str) -> ClaudeRunner:
        if not working_dir.startswith("/") and not working_dir.startswith("~"):
            working_dir = f"~/{working_dir}"
        expanded = str(Path(working_dir).expanduser().resolve())
        self.default_dir = expanded
        return self.get_session(expanded, thread_id=0)


# Global session manager
sessions = SessionManager()

# Backwards compatibility
runner = sessions.get_current_session()
