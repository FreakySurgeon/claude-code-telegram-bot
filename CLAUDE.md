# Claude Telegram Bot

Unified Telegram bot for Claude Code Remote + GTD Assistant.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI (port 8000)                         │
├─────────────────────────────────────────────────────────────────┤
│  Polling Loop (dev)      │  Polling Loop (gtd)                  │
│  Token: TELEGRAM_BOT_TOKEN│  Token: GTD_BOT_TOKEN               │
├──────────────────────────┼──────────────────────────────────────┤
│  Multi-session           │  Fixed session                       │
│  /dir, /dirs, /repos     │  GTD prompt + MCP                    │
│  Permission UI           │  Bypass permissions                  │
│  Voice → button confirm  │  Voice → direct process              │
├──────────────────────────┴──────────────────────────────────────┤
│                    ClaudeRunner (shared)                        │
│  - system_prompt injection via --append-system-prompt           │
│  - mcp_config injection via --mcp-config                        │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                     │
│  POST /webhook/email   → GTD bot processes emails               │
│  POST /cron/morning    → 6h daily briefing                      │
│  POST /cron/evening    → 18h day review                         │
│  POST /cron/weekly     → Sunday weekly review                   │
└─────────────────────────────────────────────────────────────────┘
```

## Service

```bash
# Status
sudo systemctl status claude-telegram

# Logs
sudo journalctl -u claude-telegram -f

# Restart
sudo systemctl restart claude-telegram
```

## Configuration (.env)

```env
# Dev Bot (Claude Code Remote)
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
CLAUDE_CLI_PATH=claude
CLAUDE_WORKING_DIR=/home/thomas
MODE=polling
HOST=0.0.0.0
PORT=8000
FAVORITE_REPOS=projects/revicare,projects/personal-org,media_server_docker

# GTD Bot (Personal Assistant)
GTD_BOT_TOKEN=xxx
GTD_CHAT_ID=xxx
GTD_WORKING_DIR=/home/thomas/projects/personal-org
GTD_PROMPT_PATH=/home/thomas/projects/personal-org/scripts/gtd-prompt.txt
GTD_MCP_CONFIG=/home/thomas/projects/personal-org/.mcp.json

# Transcription (both bots)
MISTRAL_API_KEY=xxx
# Local Whisper (default paths)
# WHISPER_BIN=/opt/whisper.cpp/build/bin/whisper-cli
# WHISPER_MODEL=/opt/whisper.cpp/models/ggml-medium.bin

# Email webhook
WEBHOOK_SECRET=xxx

# Hook (notify correct bot after local Claude finishes)
HOOK_SERVER_URL=http://localhost:8000
```

## Bot Behaviors

### Dev Bot (freakymexClaudeCode)

| Feature | Behavior |
|---------|----------|
| Sessions | Multi-directory (`/dir`, `/dirs`, `/repos`) |
| Permissions | Interactive UI (Allow/Deny buttons) |
| Voice | Transcribe → show text → "Send to Claude" button |
| Commands | `/start`, `/help`, `/c`, `/continue`, `/new`, `/dir`, `/dirs`, `/repos`, `/rmdir`, `/compact`, `/cancel`, `/status` |

### GTD Bot (FreakyMex-Personal-Org)

| Feature | Behavior |
|---------|----------|
| Sessions | Fixed to `GTD_WORKING_DIR` |
| Permissions | Bypass (dangerous mode) |
| Voice | Transcribe → process immediately |
| System Prompt | Loaded from `GTD_PROMPT_PATH` |
| MCP | Trello, Gmail, Calendar, GitHub, Playwright |
| Commands | `/start`, `/help`, `/new`, `/compact`, `/cancel`, `/status` |

## Cron Reminders

```crontab
0 6 * * * curl -s -X POST http://localhost:8000/cron/morning > /dev/null 2>&1
0 18 * * * curl -s -X POST http://localhost:8000/cron/evening > /dev/null 2>&1
0 10 * * 0 curl -s -X POST http://localhost:8000/cron/weekly > /dev/null 2>&1
```

Prompts are defined in `main.py:CRON_PROMPTS` dict.

## Email Webhook

Emails to `chauvet.t+claude@gmail.com` are forwarded via Google Apps Script:

```
Gmail → Apps Script (filter: label:Claude) → POST /webhook/email → GTD Bot
```

The webhook expects:
- `Authorization: Bearer $WEBHOOK_SECRET`
- JSON body with `from`, `subject`, `body`, `date`

## Voice Transcription

Uses local Whisper.cpp for audio < 5 min, Mistral Voxtral API for longer.

```python
# src/claude_telegram/transcribe.py
DURATION_THRESHOLD = 300  # seconds

async def transcribe_audio(audio_path: str) -> TranscriptionResult:
    duration = get_audio_duration(audio_path)
    if duration < DURATION_THRESHOLD:
        return transcribe_whisper(audio_path)  # Local
    else:
        return await transcribe_voxtral(audio_path)  # API
```

## Hook Notifications

When Claude finishes locally, `hook.py` notifies the correct bot based on working directory:

```bash
# ~/.claude/settings.json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python /home/thomas/projects/claude-code-telegram-bot/hook.py completed"
      }]
    }]
  }
}
```

The hook checks if `cwd` matches `GTD_WORKING_DIR` to route to the correct bot.

## Project Structure

```
claude-code-telegram-bot/
├── src/claude_telegram/
│   ├── main.py          # FastAPI, dual polling, endpoints
│   ├── config.py        # Settings (dev + gtd + transcription)
│   ├── bots.py          # BotConfig dataclass, create_bots()
│   ├── telegram.py      # Telegram API (multi-token via api_url)
│   ├── claude.py        # ClaudeRunner (system_prompt, mcp_config)
│   ├── transcribe.py    # Whisper + Voxtral
│   ├── tunnel.py        # Cloudflare tunnel
│   └── markdown.py      # MD → Telegram HTML
├── tests/
├── hook.py              # Notification hook
├── .env
└── pyproject.toml
```

## Development

```bash
# Install
uv sync

# Run locally
uv run uvicorn claude_telegram.main:app --reload

# Tests
uv run pytest -v

# Logs
tail -f /var/log/syslog | grep claude-telegram
```

## Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=claude_telegram

# Specific test
uv run pytest tests/test_handlers.py -v
```

94/95 tests pass. 1 pre-existing issue in `test_markdown.py::test_escapes_html_entities`.
