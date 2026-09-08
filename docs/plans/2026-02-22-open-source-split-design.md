# Design: Open-Source Bridge / Private GTD Split

**Date**: 2026-02-22
**Status**: Approved

## Context

The claude-code-telegram-bot project has grown into two distinct products:
1. **Bridge** — A generic Telegram ↔ Claude Code remote control (useful to any developer)
2. **GTD Bot** — A personal productivity assistant with queues, crons, email, MCP integrations (private, potential commercial product)

These need to be separated into two repositories.

## Decision

**Approach C: Two repos, bridge as git dependency**

- **Public repo**: `claude-telegram-bridge` (new repo from scratch)
- **Private repo**: Current repo continues as-is for now, GTD bot extracted later

The current running repo is NOT modified. The new public repo is built by copying and cleaning the bridge code.

## Architecture

```
~/projects/claude-telegram-bridge/     (NEW, public)
├── src/claude_telegram/
│   ├── main.py          # Bridge only (no GTD code)
│   ├── claude.py         # ClaudeRunner, SessionManager
│   ├── telegram.py       # Telegram API wrapper
│   ├── config.py         # Bridge settings only
│   ├── bots.py           # BotConfig dataclass
│   ├── topic.py          # Topic naming
│   ├── transcribe.py     # Whisper + Voxtral
│   ├── markdown.py       # MD → Telegram HTML
│   └── tunnel.py         # Cloudflare tunnel
├── hook.py               # Post-session notification hook
├── tests/                # Bridge tests only
├── LICENSE               # MIT
├── CONTRIBUTING.md
├── README.md             # User-facing documentation
├── CLAUDE.md             # Dev instructions
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml

~/projects/claude-code-telegram-bot/   (UNCHANGED, private)
└── (continues running as-is)
```

## What goes into the public bridge

### Copied as-is (no changes needed)
- `claude.py` — ClaudeRunner, SessionManager (745 lines)
- `telegram.py` — Generic Telegram API wrapper
- `topic.py` — Topic naming, Ollama integration
- `transcribe.py` — Whisper + Voxtral transcription
- `markdown.py` — Markdown to Telegram HTML
- `tunnel.py` — Cloudflare tunnel
- `hook.py` — Post-session notification

### Cleaned (GTD code removed)
- `main.py` — Remove: queue worker, cron endpoints, email webhook, calendar actions, `_load_cron_prompt()`, `_load_post_session_prompt()`, all `if bot.use_queue` conditionals
- `config.py` — Remove: `gtd_*` settings
- `bots.py` — Remove: GTD bot creation in `create_bots()`

### Not copied (GTD-only)
- `queue.py` — Entire file
- `pending_actions.py` — Entire file
- GTD-specific tests

### New files
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `README.md` (user-facing, with screenshots/features/quickstart)
- Clean `pyproject.toml` with proper metadata

## What stays in the private repo

Everything. No changes. The current repo keeps running the dual-bot setup.

Later (Phase 2, optional), the private repo can be refactored to import the bridge as a dependency instead of having its own copy. But that's a separate task.

## Credits

The public README must credit:
- Amit Mor (amimimor) — original claude-code-telegram-bot project
- Anthropic — Claude Code Remote inspiration

## Risks

- **Zero risk to current deployment** — private repo is untouched
- **Code drift** — the two repos will diverge over time. Future improvements should target the public bridge first, then be pulled into the private repo
- **No PyPI needed initially** — GTD can import via `git+https://` when ready
