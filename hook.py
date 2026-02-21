#!/usr/bin/env python3
"""
Claude Code hook script.
Called by Claude Code hooks to notify Telegram when tasks complete.

Usage:
    python hook.py completed [working_dir]
    python hook.py waiting [working_dir]

Configure in ~/.claude/settings.json:
{
  "hooks": {
    "Stop": [{"matcher": "*", "hooks": [{"type": "command", "command": "python /path/to/hook.py completed $PWD"}]}]
  }
}
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx

# Load .env from script directory
script_dir = Path(__file__).parent
env_file = script_dir / ".env"

if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Configuration
SERVER_URL = os.getenv("HOOK_SERVER_URL", "http://localhost:8000")
CLAUDE_DIR = Path.home() / ".claude"
# Number of lines to read from end of file (should be enough to find the result)
TAIL_LINES = 100


def get_project_dir(working_dir: str) -> Path | None:
    """Get the Claude project directory for a working directory."""
    projects_dir = CLAUDE_DIR / "projects"
    if not projects_dir.exists():
        return None

    # Convert path to Claude's format: /home/foo/bar -> -home-foo-bar
    # Claude also replaces dots with dashes
    abs_path = str(Path(working_dir).resolve())
    claude_dir_name = abs_path.replace("/", "-").replace(".", "-")

    project_path = projects_dir / claude_dir_name
    if project_path.exists() and project_path.is_dir():
        return project_path

    return None


def get_latest_session_file(working_dir: str) -> Path | None:
    """Find the most recent session file for a working directory."""
    project_dir = get_project_dir(working_dir)
    if not project_dir:
        return None

    sessions = [
        f for f in project_dir.glob("*.jsonl")
        if not f.name.startswith("agent-")
    ]

    if not sessions:
        return None

    return max(sessions, key=lambda f: f.stat().st_mtime)


def get_session_summary(session_file: Path, max_chars: int = 1000) -> str | None:
    """
    Extract a meaningful summary from a session file.

    Priority:
    1. The 'result' field from a 'result' type message (CLI final output)
    2. The last substantial assistant text message

    Uses tail to efficiently read only the end of large files.
    """
    if not session_file or not session_file.exists():
        return None

    try:
        # Use tail to efficiently read last N lines
        result = subprocess.run(
            ["tail", "-n", str(TAIL_LINES), str(session_file)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().split("\n")
    except Exception as e:
        print(f"Error reading session file with tail: {e}", file=sys.stderr)
        return None

    # Look for result message first, then fall back to last assistant message
    result_text = None
    last_assistant_text = None

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            msg_type = data.get("type")

            # Priority 1: result message (contains final CLI output)
            if msg_type == "result":
                result_text = data.get("result", "")

            # Priority 2: assistant message with text content
            elif msg_type == "assistant":
                content = data.get("message", {}).get("content", [])
                if isinstance(content, list):
                    texts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c.get("text", "").strip()
                            if text:
                                texts.append(text)
                    if texts:
                        # Join all text blocks from this message
                        last_assistant_text = "\n".join(texts)

        except json.JSONDecodeError:
            continue

    # Use result if available, otherwise last assistant message
    summary = result_text or last_assistant_text

    if summary:
        # Remove system tags WITH their content (internal Claude/IDE tags)
        system_tags = ['ide_opened_file', 'system-reminder', 'antml:function_calls',
                       'antml:invoke', 'antml:parameter', 'tool_result', 'ide_selection']
        for tag in system_tags:
            pattern = rf'<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>'
            summary = re.sub(pattern, '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = re.sub(rf'<{re.escape(tag)}[^>]*/>', '', summary, flags=re.IGNORECASE)

        # Remove any remaining XML/HTML tags
        summary = re.sub(r'<[^>]+>', '', summary)

        # Clean up excessive whitespace while preserving intentional line breaks
        # Replace multiple spaces with single space
        summary = re.sub(r'[ \t]+', ' ', summary)
        # Replace 3+ newlines with 2 newlines
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        # Strip leading/trailing whitespace
        summary = summary.strip()

        # Truncate if too long
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."

    return summary


def notify(event_type: str, working_dir: str | None = None):
    """Send notification to the server with optional summary."""
    summary = None
    session_id = None
    if working_dir:
        session_file = get_latest_session_file(working_dir)
        if session_file:
            summary = get_session_summary(session_file)
            session_id = session_file.stem

    try:
        response = httpx.post(
            f"{SERVER_URL}/notify/{event_type}",
            json={"summary": summary, "working_dir": working_dir, "session_id": session_id},
            timeout=10.0,
        )
        response.raise_for_status()
        print(f"Notification sent: {event_type}")
    except httpx.ConnectError:
        print(f"Could not connect to server at {SERVER_URL}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to send notification: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Skip notification if triggered by the Telegram bot to avoid duplicates
    if os.getenv("CLAUDE_TELEGRAM_BOT"):
        print("Skipping notification (triggered by Telegram bot)")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python hook.py <completed|waiting> [working_dir]", file=sys.stderr)
        sys.exit(1)

    event_type = sys.argv[1]
    working_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    notify(event_type, working_dir)
