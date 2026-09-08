# Claude Telegram Bridge — Open-Source Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a clean public repo `claude-telegram-bridge` by copying the generic bridge code from the existing private repo, removing all GTD-specific code, and preparing it for open-source publication.

**Architecture:** New repo at `~/projects/claude-telegram-bridge/` built from scratch by copying files from the existing repo. The existing repo at `~/projects/claude-code-telegram-bot/` stays untouched. No code dependencies between the two repos.

**Tech Stack:** Python 3.11+, FastAPI, httpx, pydantic-settings, uv

---

### Task 1: Initialize the new repo

**Files:**
- Create: `~/projects/claude-telegram-bridge/`
- Create: `~/projects/claude-telegram-bridge/.gitignore`
- Create: `~/projects/claude-telegram-bridge/pyproject.toml`
- Create: `~/projects/claude-telegram-bridge/src/claude_telegram/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p ~/projects/claude-telegram-bridge/src/claude_telegram
mkdir -p ~/projects/claude-telegram-bridge/tests
```

**Step 2: Create .gitignore**

Copy from existing repo:
```bash
cp ~/projects/claude-code-telegram-bot/.gitignore ~/projects/claude-telegram-bridge/
```

**Step 3: Create pyproject.toml**

```toml
[project]
name = "claude-telegram-bridge"
version = "0.1.0"
description = "Control Claude Code remotely via Telegram — multi-session, permissions UI, voice, topics, and more"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    {name = "Thomas Chauvet"},
    {name = "Amit Mor"},
]
keywords = ["claude", "telegram", "claude-code", "remote", "bridge"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Framework :: FastAPI",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "fastapi>=0.128.0",
    "httpx>=0.28.1",
    "pydantic-settings>=2.12.0",
    "python-telegram-bot>=22.5",
    "tenacity>=9.1.2",
    "uvicorn>=0.40.0",
]

[project.urls]
Repository = "https://github.com/FreakySurgeon/claude-telegram-bridge"

[project.scripts]
claude-telegram = "claude_telegram.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/claude_telegram"]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=1.3.0",
    "pytest-cov>=7.0.0",
    "respx>=0.22.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v"
pythonpath = ["src"]

[tool.coverage.run]
source = ["src/claude_telegram"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
]
```

**Step 4: Create __init__.py**

```python
"""Claude Telegram Bridge — Control Claude Code remotely via Telegram."""

__version__ = "0.1.0"
```

**Step 5: Initialize git repo**

```bash
cd ~/projects/claude-telegram-bridge
git init
git add -A
git commit -m "chore: initialize project structure"
```

---

### Task 2: Copy shared modules (unchanged)

These files are 100% generic and need zero modifications.

**Files:**
- Copy: `claude.py`, `telegram.py`, `topic.py`, `transcribe.py`, `markdown.py`, `tunnel.py`

**Step 1: Copy the files**

```bash
cd ~/projects/claude-code-telegram-bot/src/claude_telegram
for f in claude.py telegram.py topic.py transcribe.py markdown.py tunnel.py; do
    cp "$f" ~/projects/claude-telegram-bridge/src/claude_telegram/
done
```

**Step 2: Copy hook.py**

```bash
cp ~/projects/claude-code-telegram-bot/hook.py ~/projects/claude-telegram-bridge/
```

**Step 3: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add -A
git commit -m "feat: add shared modules (claude runner, telegram API, transcription, markdown, tunnel, hook)"
```

---

### Task 3: Create clean config.py (remove GTD settings)

**Files:**
- Create: `~/projects/claude-telegram-bridge/src/claude_telegram/config.py`

**Step 1: Write config.py**

Copy from existing `config.py` but remove all `gtd_*` fields and `webhook_secret`:

```python
"""Configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str  # Allowed chat ID for security

    # Claude
    claude_cli_path: str = "claude"
    claude_working_dir: str | None = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    webhook_path: str = "/webhook"
    webhook_url: str | None = None  # Manual webhook URL (for "webhook" mode)

    # Mode: "polling" (default), "tunnel", or "webhook"
    mode: str = "polling"

    # Favorite repos (comma-separated paths relative to home)
    favorite_repos: str = ""

    # Transcription
    mistral_api_key: str | None = None
    whisper_bin: str = "/opt/whisper.cpp/build/bin/whisper-cli"
    whisper_model: str = "/opt/whisper.cpp/models/ggml-medium.bin"

    # Hook server URL (for hook.py notifications)
    hook_server_url: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_favorite_repos(self) -> list[str]:
        """Parse favorite repos from comma-separated string."""
        if not self.favorite_repos:
            return []
        return [r.strip() for r in self.favorite_repos.split(",") if r.strip()]


settings = Settings()
```

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add src/claude_telegram/config.py
git commit -m "feat: add config (bridge-only settings, no GTD)"
```

---

### Task 4: Create clean bots.py (dev bot only)

**Files:**
- Create: `~/projects/claude-telegram-bridge/src/claude_telegram/bots.py`

**Step 1: Write bots.py**

Copy from existing but remove GTD bot creation:

```python
"""Bot configuration — single dev bot for Claude Code bridge."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for a Telegram bot identity."""
    name: str
    token: str
    chat_id: str
    fixed_working_dir: str | None = None
    system_prompt_path: str | None = None
    mcp_config_path: str | None = None
    use_queue: bool = False
    commands_whitelist: list[str] = field(default_factory=list)

    username: str | None = None  # Populated at startup via getMe

    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    @property
    def system_prompt(self) -> str | None:
        if not self.system_prompt_path:
            return None
        try:
            return Path(self.system_prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read system prompt {self.system_prompt_path}: {e}")
            return None

    def is_authorized(self, chat_id: str | int) -> bool:
        return str(chat_id) == str(self.chat_id)


def create_bots() -> dict[str, BotConfig]:
    """Create bot configurations from settings."""
    bots = {}

    bots["dev"] = BotConfig(
        name="dev",
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        use_queue=False,
        commands_whitelist=[
            "/start", "/help", "/c", "/continue", "/new", "/resume", "/dir", "/dirs",
            "/repos", "/rmdir", "/compact", "/cancel", "/status",
        ],
    )

    return bots
```

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add src/claude_telegram/bots.py
git commit -m "feat: add bots config (dev bot only)"
```

---

### Task 5: Create clean main.py (remove all GTD code)

This is the biggest task. We copy `main.py` and surgically remove:
- Lines 37-65: `_load_cron_prompt()`, `_load_post_session_prompt()`
- Line 82: import of `queue.py` and `pending_actions.py`
- Lines 108-112: GTD queue globals
- Lines 239-261: Queue init/cleanup in lifespan
- Lines 279-322: `_replay_persistent_queue()`, `queue_worker()`
- Lines 336-338: Queue size in `/health`
- Lines 406-428: Queue branching in `handle_message()`
- Lines 657-681: Voice queue branching
- Lines 754-779: Photo queue branching
- Lines 830-869: GTD help text in `/start`
- Lines 1034-1035: Queue drain in `/cancel`
- Lines 1052-1053: Queue status in `/status`
- Line 1375: `if bot.use_queue: bypass_permissions = True`
- Lines 1753-2106: All GTD endpoints (email webhook, calendar actions, cron)

**Files:**
- Create: `~/projects/claude-telegram-bridge/src/claude_telegram/main.py`

**Step 1: Copy the file**

```bash
cp ~/projects/claude-code-telegram-bot/src/claude_telegram/main.py \
   ~/projects/claude-telegram-bridge/src/claude_telegram/main.py
```

**Step 2: Remove GTD imports (top of file)**

Remove lines 82-88 (imports of queue, pending_actions):
```python
# REMOVE these lines:
from .queue import QueueItem, RequestQueue, process_queue_item, PersistentQueue, ApiStatus
from .pending_actions import (
    add_action,
    cleanup_actions,
    is_duplicate,
)
```

**Step 3: Remove GTD helper functions (lines 37-65)**

Remove `_load_cron_prompt()` and `_load_post_session_prompt()` entirely.

**Step 4: Remove GTD queue globals (lines 108-112)**

Remove:
```python
# Queue for GTD bot (initialized in lifespan)
gtd_queue: RequestQueue | None = None
queue_worker_task: asyncio.Task | None = None
persistent_queue: PersistentQueue | None = None
api_status: ApiStatus | None = None
```

**Step 5: Clean lifespan() — remove queue init and cleanup**

Remove lines 239-251 (queue startup) and lines 256-261 (queue cleanup).

In the global declaration on line 184, remove `gtd_queue, queue_worker_task, persistent_queue, api_status`.

**Step 6: Remove queue functions**

Remove `_replay_persistent_queue()` (lines 279-300) and `queue_worker()` (lines 303-322) entirely.

**Step 7: Clean /health endpoint**

Remove queue_size and persistent_queue_size and api_unavailable from the health dict:
```python
# REMOVE:
"queue_size": gtd_queue.size if gtd_queue else 0,
"persistent_queue_size": persistent_queue.size if persistent_queue else 0,
"api_unavailable": api_status.unavailable if api_status else False,
```

**Step 8: Clean handle_message() — remove queue branching**

Remove lines 406-428 (the entire `if bot.use_queue and gtd_queue is not None:` block).
The function should flow directly to `run_claude()`.

**Step 9: Clean handle_voice() — remove queue branching**

Remove lines 657-682 (the `if bot.use_queue:` block). Keep only the dev bot path (show transcription with button).

**Step 10: Clean handle_photo() — remove queue branching**

Remove lines 754-779 (the `if bot.use_queue and gtd_queue is not None:` block). Keep only the dev bot direct execution path.

**Step 11: Clean handle_command() — remove GTD help text**

In `/start` and `/help`: remove the `if not bot.use_queue:` / `else:` conditional. Keep only the dev bot help text.

**Step 12: Clean /cancel — remove queue drain**

Remove:
```python
if gtd_queue and bot.use_queue:
    drained = gtd_queue.drain()
```
And simplify the message (remove drained count).

**Step 13: Clean /status — remove queue display**

Remove:
```python
if gtd_queue and bot.use_queue:
    msg += f"\n📥 Queue: {gtd_queue.size} en attente"
```

**Step 14: Clean run_claude() — remove bypass for queue**

Remove:
```python
if bot.use_queue:
    bypass_permissions = True
```

**Step 15: Remove all GTD endpoints**

Remove everything from line 1753 to line 2106:
- `email_webhook()` + `_process_email()`
- `cron_calendar_actions()` + `_process_calendar_actions()`
- `cron_reminder()` + `_process_cron()`

**Step 16: Verify the file loads**

```bash
cd ~/projects/claude-telegram-bridge
uv sync
TELEGRAM_BOT_TOKEN=test TELEGRAM_CHAT_ID=test uv run python -c "from claude_telegram.main import app; print('OK')"
```

Expected: `OK`

**Step 17: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add src/claude_telegram/main.py
git commit -m "feat: add main.py (bridge only, all GTD code removed)"
```

---

### Task 6: Copy and adapt tests

**Files:**
- Copy: `conftest.py`, `test_main.py`, `test_claude.py`, `test_telegram.py`, `test_topic.py`, `test_markdown.py`, `test_tunnel.py`, `test_hook.py`
- Skip: `test_queue.py`, `test_pending_actions.py` (GTD-only)

**Step 1: Copy test files**

```bash
cd ~/projects/claude-code-telegram-bot/tests
for f in conftest.py __init__.py test_main.py test_claude.py test_telegram.py test_topic.py test_markdown.py test_tunnel.py test_hook.py; do
    cp "$f" ~/projects/claude-telegram-bridge/tests/
done
```

**Step 2: Clean conftest.py**

Remove GTD env vars (lines 13-17):
```python
# REMOVE:
os.environ.setdefault("GTD_BOT_TOKEN", "")
os.environ.setdefault("GTD_CHAT_ID", "")
os.environ.setdefault("GTD_WORKING_DIR", "")
os.environ.setdefault("GTD_PROMPT_PATH", "")
os.environ.setdefault("GTD_MCP_CONFIG", "")
```

Remove the `WEBHOOK_SECRET` env var too.

Remove the `gtd_bot` fixture (lines 81-94).

Remove GTD settings from `mock_settings` fixture:
```python
# REMOVE:
mock.gtd_bot_token = None
mock.gtd_chat_id = None
mock.gtd_working_dir = None
mock.gtd_prompt_path = None
mock.gtd_mcp_config = None
mock.webhook_secret = None
```

**Step 3: Clean test_main.py**

Remove any tests that reference `gtd_queue`, `QueueItem`, `use_queue`, `gtd_bot`, email webhook, cron endpoints. Read the file first and identify which tests to remove.

**Step 4: Run tests**

```bash
cd ~/projects/claude-telegram-bridge
uv run pytest -v
```

Fix any import errors or test failures caused by removed GTD code.

**Step 5: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add tests/
git commit -m "test: add bridge tests (GTD tests excluded)"
```

---

### Task 7: Copy Docker and service files

**Files:**
- Copy: `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh` (if exists)
- Create: `.env.example`
- Copy: `claude-telegram.service`

**Step 1: Copy Docker files**

```bash
cp ~/projects/claude-code-telegram-bot/Dockerfile ~/projects/claude-telegram-bridge/
cp ~/projects/claude-code-telegram-bot/docker-compose.yml ~/projects/claude-telegram-bridge/
[ -f ~/projects/claude-code-telegram-bot/docker-entrypoint.sh ] && cp ~/projects/claude-code-telegram-bot/docker-entrypoint.sh ~/projects/claude-telegram-bridge/
```

**Step 2: Create .env.example**

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Claude Configuration
CLAUDE_CLI_PATH=claude
CLAUDE_WORKING_DIR=/path/to/your/project

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Mode: "polling" (default), "tunnel" (auto-creates public URL), or "webhook" (manual URL)
MODE=polling

# Webhook settings (only needed if MODE=webhook)
# WEBHOOK_PATH=/webhook
# WEBHOOK_URL=https://your-public-url.com

# Favorite repos — comma-separated, relative to ~ (optional)
# FAVORITE_REPOS=projects/foo,projects/bar

# Voice transcription (optional)
# MISTRAL_API_KEY=your_mistral_key
# WHISPER_BIN=/opt/whisper.cpp/build/bin/whisper-cli
# WHISPER_MODEL=/opt/whisper.cpp/models/ggml-medium.bin

# Hook Configuration (for hook.py notifications)
# HOOK_SERVER_URL=http://localhost:8000
```

**Step 3: Copy service file**

```bash
cp ~/projects/claude-code-telegram-bot/claude-telegram.service ~/projects/claude-telegram-bridge/
```

Update the `WorkingDirectory` path inside to use a generic path (the user will adjust).

**Step 4: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add -A
git commit -m "chore: add Docker, systemd service, and .env.example"
```

---

### Task 8: Create LICENSE file

**Files:**
- Create: `~/projects/claude-telegram-bridge/LICENSE`

**Step 1: Write MIT license**

```
MIT License

Copyright (c) 2026 Thomas Chauvet, Amit Mor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add LICENSE
git commit -m "chore: add MIT license"
```

---

### Task 9: Create CONTRIBUTING.md

**Files:**
- Create: `~/projects/claude-telegram-bridge/CONTRIBUTING.md`

**Step 1: Write CONTRIBUTING.md**

```markdown
# Contributing to Claude Telegram Bridge

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo
2. Clone your fork: `git clone https://github.com/your-username/claude-telegram-bridge.git`
3. Install dependencies: `uv sync`
4. Create a branch: `git checkout -b feat/my-feature`

## Development

```bash
# Run locally
uv run uvicorn claude_telegram.main:app --reload

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=claude_telegram
```

## Pull Requests

- Keep PRs focused on a single change
- Add tests for new features
- Follow existing code style
- Update README if adding user-facing features

## Reporting Issues

Open an issue with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and OS
```

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md"
```

---

### Task 10: Write the public README

**Files:**
- Create: `~/projects/claude-telegram-bridge/README.md`

**Step 1: Write README.md**

Use the existing README as a base but:
- Update the repo URL to `claude-telegram-bridge`
- Add a proper "Features" section with a feature list
- Add proper attribution section at the bottom
- Remove any GTD references
- Add "Voice messages" and "Photo analysis" and "Forum topics" to features
- Ensure Quick Start defaults to polling mode (simpler)
- Add a "Hooks" section explaining hook.py

The README should already exist in the existing repo and is mostly correct. Copy and refine:
- Title: "Claude Telegram Bridge"
- Subtitle: "Control Claude Code remotely via Telegram"
- Attribution at bottom: credits to Amit Mor and Anthropic

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add README.md
git commit -m "docs: add README with features, quickstart, and attribution"
```

---

### Task 11: Create CLAUDE.md for the public repo

**Files:**
- Create: `~/projects/claude-telegram-bridge/CLAUDE.md`

**Step 1: Write CLAUDE.md**

A simpler version of the existing CLAUDE.md, without GTD references:

```markdown
# Claude Telegram Bridge

Telegram bot that bridges Claude Code for remote development.

## Architecture

Single bot (dev) with FastAPI server, polling/tunnel/webhook modes.

## Development

```bash
uv sync
uv run uvicorn claude_telegram.main:app --reload
uv run pytest -v
```

## Project Structure

```
src/claude_telegram/
├── main.py          # FastAPI app, message handlers, commands
├── config.py        # Pydantic settings from .env
├── bots.py          # BotConfig dataclass
├── claude.py        # ClaudeRunner, SessionManager
├── telegram.py      # Telegram API wrapper
├── topic.py         # Forum topic naming
├── transcribe.py    # Whisper + Voxtral transcription
├── markdown.py      # MD → Telegram HTML
└── tunnel.py        # Cloudflare tunnel
```

## Tests

```bash
uv run pytest -v
uv run pytest --cov=claude_telegram
```
```

**Step 2: Commit**

```bash
cd ~/projects/claude-telegram-bridge
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md for development"
```

---

### Task 12: Final verification

**Step 1: Run full test suite**

```bash
cd ~/projects/claude-telegram-bridge
uv run pytest -v --cov=claude_telegram
```

All tests must pass.

**Step 2: Verify import works**

```bash
TELEGRAM_BOT_TOKEN=test TELEGRAM_CHAT_ID=test uv run python -c "
from claude_telegram.main import app
from claude_telegram.claude import ClaudeRunner
from claude_telegram.telegram import send_message
from claude_telegram.config import Settings
print('All imports OK')
"
```

**Step 3: Verify no GTD references remain**

```bash
cd ~/projects/claude-telegram-bridge
grep -ri "gtd" src/ tests/ --include="*.py" || echo "No GTD references found"
grep -ri "pending_actions" src/ tests/ --include="*.py" || echo "No pending_actions references"
grep -ri "queue_worker" src/ tests/ --include="*.py" || echo "No queue_worker references"
grep -ri "cron_reminder" src/ tests/ --include="*.py" || echo "No cron references"
grep -ri "email_webhook" src/ tests/ --include="*.py" || echo "No email_webhook references"
```

All should report "No ... references found".

**Step 4: Verify no secrets in files**

```bash
cd ~/projects/claude-telegram-bridge
grep -ri "chauvet\|freakymex\|personal-org\|192\.168\.\|100\.108\." src/ tests/ *.md --include="*.py" --include="*.md" || echo "No personal info found"
```

**Step 5: Check git log looks clean**

```bash
cd ~/projects/claude-telegram-bridge
git log --oneline
```

**Step 6: Commit any final fixes**

```bash
cd ~/projects/claude-telegram-bridge
git add -A
git status
# If changes exist:
git commit -m "fix: final cleanup for open-source publication"
```

---

### Task 13: Verify existing repo still works

**Step 1: Run tests on the existing repo**

```bash
cd ~/projects/claude-code-telegram-bot
uv run pytest -v
```

All 94 tests should still pass (we didn't modify anything).

**Step 2: Check service status**

```bash
sudo systemctl status claude-telegram
```

Should still be running.

---

## Summary of what was created

| File | Source | Changes |
|------|--------|---------|
| `pyproject.toml` | New | Clean metadata, MIT license, no GTD deps |
| `config.py` | Copied | Removed `gtd_*` settings, `webhook_secret` |
| `bots.py` | Copied | Removed GTD bot creation |
| `main.py` | Copied | Removed ~600 lines of GTD code (queue, cron, email, calendar) |
| `claude.py` | Copied | Unchanged |
| `telegram.py` | Copied | Unchanged |
| `topic.py` | Copied | Unchanged |
| `transcribe.py` | Copied | Unchanged |
| `markdown.py` | Copied | Unchanged |
| `tunnel.py` | Copied | Unchanged |
| `hook.py` | Copied | Unchanged |
| `tests/` | Copied | Removed GTD tests and fixtures |
| `LICENSE` | New | MIT |
| `README.md` | Rewritten | User-facing, no GTD references |
| `CLAUDE.md` | New | Dev instructions for bridge only |
| `CONTRIBUTING.md` | New | Contributor guidelines |
| `.env.example` | New | Bridge-only settings |
| `Dockerfile` | Copied | Unchanged |
| `docker-compose.yml` | Copied | Unchanged |
