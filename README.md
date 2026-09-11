# Claude Telegram

Control Claude Code remotely via Telegram. A Python/FastAPI bridge that lets you interact with Claude Code from anywhere.

> **Attribution**: Inspired by [Claude-Code-Remote](https://github.com/anthropics/Claude-Code-Remote), reimplemented with a cleaner Python architecture.

## Features

- **Multi-session support** - Run Claude in different directories simultaneously
- **Permission handling** - Approve or deny Claude's permission requests via Telegram buttons
- **Animated status messages** - Rotating "Thinking...", "Pondering..." etc. while Claude works
- **Auto-continue conversations** - Just reply naturally, no commands needed
- **Quick-reply buttons** - Tap numbered options directly
- **Markdown rendering** - Claude's markdown converted to Telegram HTML
- **Smart session handling** - 10-minute auto-continue window per session
- **Three connection modes** - Tunnel (default), Polling, or Webhook

## Quick Start

### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Choose a name (e.g., "My Claude Bot")
4. Choose a username (must end in `bot`, e.g., `my_claude_bot`)
5. **Save the bot token** - looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### 2. Get Your Chat ID

1. Start a chat with your new bot (search for it by username)
2. Send any message to it (e.g., "hello")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Find `"chat":{"id":` in the response - that number is your chat ID
   - Example: `"chat":{"id":123456789` → your chat ID is `123456789`

### 3. Install Prerequisites

```bash
# Clone the repo
git clone https://github.com/yourusername/claude-telegram.git
cd claude-telegram

# Install Python dependencies
uv sync

# Install cloudflared (for tunnel mode)
# macOS:
brew install cloudflared

# Linux:
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
CLAUDE_CLI_PATH=claude
CLAUDE_WORKING_DIR=/path/to/your/project
MODE=polling
```

### 5. Run

```bash
uv run uvicorn claude_telegram.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
Starting Cloudflare tunnel...
Tunnel started: https://random-words.trycloudflare.com
Setting webhook...
Webhook set successfully
Application startup complete.
```

Now send a message to your bot!

## Connection Modes

### Polling Mode (Default)

No public URL needed - polls Telegram's servers directly. Simple and reliable.

```bash
uv run uvicorn claude_telegram.main:app
```

### Tunnel Mode

Uses Cloudflare's free quick tunnels to create a public URL automatically. Lower latency but requires DNS propagation (2-5 min on first start).

```bash
MODE=tunnel uv run uvicorn claude_telegram.main:app
```

### Webhook Mode

Use your own public URL (e.g., behind nginx, Caddy, or a cloud provider).

```bash
MODE=webhook WEBHOOK_URL=https://your-domain.com uv run uvicorn claude_telegram.main:app
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start`, `/help` | Show help with formatted commands |
| `/c <message>` | Continue previous session |
| `/new <message>` | Start fresh session (reset context) |
| `/dir <path>` | Switch to a different directory/session |
| `/dirs` | List all active sessions |
| `/rmdir <path>` | Remove a session from the list |
| `/compact` | Compact conversation context |
| `/cancel` | Cancel current running task |
| `/status` | Check if Claude is running |
| `<any text>` | Auto-continues if within 10 min, else new session |

**Tips:**
- Just type naturally - conversations auto-continue for 10 minutes
- Quick replies like "1", "2", "yes", "no" always continue
- Tap inline buttons for numbered options

## Multi-Directory Sessions

Run Claude in different project directories simultaneously. Each directory maintains its own conversation context and history.

> **⚠️ Important:** Avoid using `/dir` on directories where you're actively running Claude locally. The bot and local CLI share the same session files (`~/.claude/projects/`), which can cause conflicts and unexpected behavior. Use the bot for directories you're not working on locally, or close your local Claude session first.

**Add a new directory:**
```
/dir ~/projects/backend
```

**Switch between sessions:** Use `/dirs` to see all sessions with numbered buttons:
```
You: /dirs
Bot: Active Sessions
     → 1. 💤 frontend
       2. 💤 api
     [✓ 1. frontend] [2. api]   ← tap to switch!
```

**Example workflow:**
```
You: /dir ~/projects/api
Bot: 📂 Switched to api
     Status: 💤 idle • fresh

You: add input validation to the user endpoint
Bot: [api] 🧠 Thinking...
Bot: I'll add validation to src/routes/user.ts...

You: /dir ~/projects/frontend
Bot: 📂 Switched to frontend

You: /dirs
Bot: Active Sessions
     → 1. 💤 frontend
       2. 💤 api
     [✓ 1. frontend] [2. api]

You: *taps [2. api] button*
Bot: 📂 Switched to api
     Status: 💤 idle • in conversation
```

Each session:
- **Resumes from stored Claude sessions** - picks up where you left off using `~/.claude/projects/`
- **Shows previous context** - displays your last 5 messages when switching to a stored session
- Has its own 10-minute auto-continue window
- Maintains separate conversation context
- Shows directory name in status messages (e.g., `[api] Thinking...`)
- Quick-switch via numbered buttons

## Permission Handling

When Claude needs to perform actions requiring permission (writing files, running commands, etc.), you'll get an interactive prompt in Telegram:

```
You: create a file /tmp/hello.txt with hello world

Bot: ⚠️ Permission denied:
     • Write to /tmp/hello.txt

     I need write permission for `/tmp/hello.txt`.

     [✅ Allow & Retry] [❌ Deny]
```

- **Allow & Retry** - Grants permission and retries the action
- **Deny** - Cancels the request

The bot captures permission denials from Claude's output and presents them as actionable buttons, so you can approve operations remotely without direct terminal access.

**Supported permission types:**
- `Write` - Creating or overwriting files
- `Edit` - Modifying existing files
- `Read` - Reading files outside the working directory
- `Bash` - Running shell commands

**Context preview when switching:**
```
You: /dir projects/api
Bot: 📂 Switched to api
     Status: 💤 idle • fresh

     📜 Previous context:
     • add input validation to the user endpoint
     • fix the failing tests in auth module
     • update the API documentation
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | (required) | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | (required) | Your chat ID (security: only this chat can use the bot) |
| `CLAUDE_CLI_PATH` | `claude` | Path to Claude CLI |
| `CLAUDE_WORKING_DIR` | (none) | Working directory for Claude |
| `MODE` | `polling` | `polling`, `tunnel`, or `webhook` |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `WEBHOOK_URL` | (none) | Your public URL (webhook mode only) |
| `CHANNEL_ROUTING_PATH` | (none) | YAML file with a `channels:` block (see `routing.example.yaml`). Unset = everything goes to Telegram |
| `CHANNEL_ENV_FILE` | (none) | Optional dotenv file used to expand `${VAR}` placeholders in the routing file |

## Channels (ports & adapters)

The core (`ports.py`, `routing.py`, `notifications.py`, `conversations.py`) is
channel-agnostic; Telegram and Zulip are adapters under `adapters/`. No module
outside `adapters/` imports Telegram code.

- **Outbound**: crons, webhooks and `/notify/{event_type}` build an `Event`
  (type, severity, title, body). The `RoutingPolicy` sends `urgent` events to
  every channel listed in `urgent` and the rest to `default`; a channel's
  `only_severity` drops everything else for that channel.
- **Inbound (Zulip)**: an event-queue long-poll listens to `listen_streams`
  (every message), `mention_streams` (on @mention) and DMs. One topic = one
  Claude session (idle TTL `session_ttl_hours`). The event cursor is stored in
  `$DATA_DIR/zulip-events.json`.
- **Webhook fallback**: `POST /webhook/zulip` (Zulip outgoing webhook) is kept
  as a safety net. When the event queue runs, a payload is only processed if
  the queue has not seen the message after 2 minutes.
- **Local testing**: `POST /channels/inject` (loopback + `X-Webhook-Secret`)
  publishes an `Event` or feeds an inbound message through the core.

## Docker

```bash
# Build
docker build -t claude-telegram .

# Run with tunnel (default)
docker run -d \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  -v /usr/local/bin/claude:/usr/local/bin/claude:ro \
  -v $(pwd):/workspace \
  claude-telegram

# Run with polling
docker run -d \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  -e MODE=polling \
  -v /usr/local/bin/claude:/usr/local/bin/claude:ro \
  -v $(pwd):/workspace \
  claude-telegram
```

## Claude Code Hooks

Get notified in Telegram when Claude finishes:

**Option 1: Environment variable**
```bash
export CLAUDE_HOOKS_CONFIG=/path/to/claude-telegram/claude-hooks.json
claude
```

**Option 2: Add to `~/.claude/settings.json`**
```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python /path/to/claude-telegram/hook.py completed",
        "timeout": 30000
      }]
    }]
  }
}
```

## Troubleshooting

### "Webhook setup failed" / DNS errors

This is normal for tunnel mode! Cloudflare quick tunnels take 2-5 minutes for DNS to propagate globally. The app retries automatically with exponential backoff (up to 15 attempts). Just wait.

### Bot doesn't respond

1. Check the chat ID matches your `.env`
2. Make sure you messaged the bot first (it can't initiate)
3. Check server logs for errors

### "Claude is busy"

Claude is still processing. Use `/cancel` to stop it, or wait.

## Development

```bash
# Run tests
uv run pytest -v --cov=claude_telegram

# Run with reload
uv run uvicorn claude_telegram.main:app --reload
```

## Project Structure

```
claude-telegram/
├── src/claude_telegram/
│   ├── main.py          # FastAPI app (composition root, endpoints)
│   ├── config.py        # Pydantic settings
│   ├── ports.py         # Channel-agnostic types (Event, InboundMessage, ...)
│   ├── routing.py       # RoutingPolicy (channels: block)
│   ├── notifications.py # NotificationService (publish/reply)
│   ├── conversations.py # ConversationService (inbound → Claude session)
│   ├── adapters/
│   │   ├── telegram/    # Telegram API, handlers, outbound
│   │   └── zulip/       # Zulip client, event-queue inbound, outbound
│   ├── claude.py        # Claude CLI runner
│   ├── tunnel.py        # Cloudflare Tunnel manager
│   └── markdown.py      # MD → Telegram HTML
├── tests/               # Pytest tests
├── hook.py              # Hook notification script
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
