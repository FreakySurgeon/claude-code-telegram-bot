"""Channel adapters (Telegram, Zulip, ...) and their composition."""

from __future__ import annotations

import logging
from pathlib import Path

from ..bots import BotConfig
from ..ports import OutboundChannel
from ..routing import RoutingPolicy

logger = logging.getLogger(__name__)

ZULIP_STATE_FILE = "zulip-events.json"


def zulip_client_from(policy: RoutingPolicy):
    """ZulipClient built from the routing policy, or None when Zulip is not configured."""
    cfg = policy.channel_config("zulip")
    if not (cfg.get("site") and cfg.get("api_key") and cfg.get("bot_email")):
        return None
    from .zulip.client import ZulipClient

    return ZulipClient(cfg["site"], cfg["bot_email"], cfg["api_key"], host_header=cfg.get("host_header") or None)


def build_outbounds(
    policy: RoutingPolicy, bots: dict[str, BotConfig], data_dir: Path, zulip_client=None
) -> dict[str, OutboundChannel]:
    """One OutboundChannel per configured channel (Telegram is always available)."""
    from .telegram.outbound import TelegramOutbound

    outbounds: dict[str, OutboundChannel] = {"telegram": TelegramOutbound(bots)}
    client = zulip_client or zulip_client_from(policy)
    if client is not None:
        from .zulip.outbound import OwnedTopics, ZulipOutbound

        outbounds["zulip"] = ZulipOutbound(client, owned=OwnedTopics(Path(data_dir) / ZULIP_STATE_FILE))
    elif "zulip" in policy.channels:
        logger.warning("Zulip routed but not configured (site/bot_email/api_key missing)")
    return outbounds
