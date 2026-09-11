"""RoutingPolicy: decides which channels receive an Event.

The policy is read from the ``channels`` key of a YAML file designated by
``CHANNEL_ROUTING_PATH`` (see ``routing.example.yaml``). Without a file,
everything goes to Telegram (the historical behaviour).

``${VAR}`` placeholders are expanded from the process environment merged with
an optional dotenv file (``CHANNEL_ENV_FILE``) so credentials are read where
they already live instead of being copied; the dotenv values are never
written into ``os.environ``.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values

from .ports import Event

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

DEFAULT_CONFIG: dict = {"default": "telegram", "urgent": ["telegram"]}

# Cron/pipeline name -> event type (the keys of `outputs` in the routing file).
PIPELINE_EVENT_TYPES: dict[str, str] = {
    "morning": "briefing_morning",
    "evening": "briefing_evening",
    "whatsapp": "whatsapp_triage",
    "gdrive-inbox": "gdrive_inbox",
    "agent-tasks": "agent_tasks",
    "weekly": "weekly",
    "sent-emails": "sent_emails",
    "garmin-sync": "garmin",
    "enrichment": "enrichment",
    "limitless": "limitless",
    "omi": "omi",
    "zulip": "zulip_fallback",
    "calendar-action": "calendar_action",
}


def event_type_for(reminder_type: str) -> str:
    return PIPELINE_EVENT_TYPES.get(reminder_type, reminder_type.replace("-", "_"))


@dataclass(frozen=True)
class ChannelRef:
    """A channel plus a channel-specific target (stream/topic, bot, …)."""

    channel: str
    target: dict = field(default_factory=dict, hash=False)


def expand_env(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: env.get(m.group(1)) or "", value)
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    return value


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_target(target: dict, event: Event) -> dict:
    values = _SafeDict(date=dt.date.today().isoformat(), title=event.title or event.type)
    return {k: (v.format_map(values) if isinstance(v, str) else v) for k, v in target.items()}


class RoutingPolicy:
    def __init__(self, config: dict | None):
        self.config = dict(config or DEFAULT_CONFIG)

    @classmethod
    def default(cls) -> "RoutingPolicy":
        return cls(DEFAULT_CONFIG)

    @classmethod
    def from_file(cls, path: str | Path | None, env_file: str | Path | None = None) -> "RoutingPolicy":
        if not path:
            return cls.default()
        path = Path(path)
        if not path.is_file():
            logger.warning("Routing file %s not found, using default policy (Telegram)", path)
            return cls.default()
        raw = yaml.safe_load(path.read_text()) or {}
        channels = raw.get("channels")
        if not channels:
            logger.warning("No 'channels' key in %s, using default policy (Telegram)", path)
            return cls.default()
        env: dict[str, str] = {}
        if env_file and Path(env_file).is_file():
            env.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        env.update(os.environ)
        return cls(expand_env(channels, env))

    @property
    def default_channel(self) -> str:
        return self.config.get("default", "telegram")

    @property
    def channels(self) -> set[str]:
        names = {self.default_channel, *self.config.get("urgent", [])}
        names.update(k for k, v in self.config.items() if isinstance(v, dict))
        return names

    def channel_config(self, name: str) -> dict:
        cfg = self.config.get(name)
        return cfg if isinstance(cfg, dict) else {}

    def _target(self, channel: str, event: Event) -> dict:
        cfg = self.channel_config(channel)
        if channel == "telegram":
            return {"bot": cfg.get("bot", "gtd")}
        outputs = cfg.get("outputs") or {}
        target = outputs.get(event.type) or outputs.get("default") or {}
        return _format_target(dict(target), event)

    def route(self, event: Event) -> list[ChannelRef]:
        if event.type.startswith("dev."):
            allowed = self.channel_config("telegram").get("dev_bot_events")
            if allowed is None or event.type[4:] in allowed:
                return [ChannelRef("telegram", {"bot": "dev"})]
            return []

        if event.severity == "urgent":
            candidates = self.config.get("urgent") or [self.default_channel]
        else:
            candidates = [self.default_channel]

        routes: list[ChannelRef] = []
        seen: set[str] = set()
        for channel in candidates:
            if channel in seen:
                continue
            seen.add(channel)
            only = self.channel_config(channel).get("only_severity")
            if only and event.severity not in only:
                continue
            routes.append(ChannelRef(channel, self._target(channel, event)))
        return routes
