"""Shared application state.

Set by the composition root (``main.lifespan``) and read by the channel
adapters and core services, so none of them has to import ``main``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bots import BotConfig
    from .notifications import NotificationService
    from .ports import ConversationRef
    from .queue import ApiStatus, PersistentQueue, RequestQueue

# Bot configurations (initialized at startup)
bots: "dict[str, BotConfig]" = {}
# Map chat_id -> bot_name for routing notifications
chat_to_bot: dict[str, str] = {}

# GTD queue (initialized in lifespan)
gtd_queue: "RequestQueue | None" = None
queue_worker_task: asyncio.Task | None = None
persistent_queue: "PersistentQueue | None" = None
api_status: "ApiStatus | None" = None

# Channel services (initialized in lifespan)
notifications: "NotificationService | None" = None
conversations: Any = None   # ConversationService, once a non-Telegram inbound channel runs
session_store: Any = None   # SessionStore for non-Telegram conversations
inbounds: list[Any] = []    # running InboundChannel instances
inbound_tasks: list[Any] = []  # asyncio tasks running InboundChannel.run
zulip_inbound: Any = None   # ZulipInbound when the Zulip event queue runs (webhook becomes a fallback)

# Telegram UI state
pending_permissions: dict[str, dict] = {}  # chat_id -> {message, denials, session_key, bot_name}
pending_voice_texts: dict[str, str] = {}   # chat_id -> full transcription text
resume_working_dirs: dict[str, str] = {}   # session_id -> working_dir
# job_id -> {session_id, task, thread_id, provider_config, status}
pending_computer_use: dict[str, dict] = {}
polling_tasks: list[asyncio.Task] = []
tunnel_url: str | None = None


def bot_for_ref(ref: "ConversationRef") -> "BotConfig | None":
    """Bot owning a conversation: explicit ``ref.bot``, else the bot of the chat."""
    if ref.bot and ref.bot in bots:
        return bots[ref.bot]
    name = chat_to_bot.get(str(ref.conversation_id))
    return bots.get(name) if name else None
