"""NotificationService: the core's only way to talk to the user.

Producers (crons, webhooks, queue) build an ``Event``; the RoutingPolicy picks
the channels; each channel's OutboundChannel delivers it. A failing channel is
logged and reported, never raised, so one broken chat API cannot block others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from .ports import Action, Attachment, ConversationRef, Event, OutboundChannel
from .routing import RoutingPolicy

logger = logging.getLogger(__name__)


@dataclass
class Delivery:
    channel: str
    ref: ConversationRef | None = None
    message_ids: list[str] = field(default_factory=list)
    error: str | None = None


class NotificationService:
    def __init__(self, policy: RoutingPolicy, outbounds: dict[str, OutboundChannel] | None = None):
        self.policy = policy
        self._outbounds: dict[str, OutboundChannel] = dict(outbounds or {})

    def register(self, name: str, outbound: OutboundChannel) -> None:
        self._outbounds[name] = outbound

    def outbound(self, channel: str) -> OutboundChannel | None:
        return self._outbounds.get(channel)

    async def publish(self, event: Event) -> list[Delivery]:
        deliveries: list[Delivery] = []
        for route in self.policy.route(event):
            out = self._outbounds.get(route.channel)
            if out is None:
                logger.warning("No outbound for channel %s (event %s)", route.channel, event.type)
                deliveries.append(Delivery(route.channel, error="channel not configured"))
                continue
            ref = None
            try:
                hint = event.conversation_hint
                if hint is not None and hint.channel == route.channel:
                    ref = hint
                else:
                    ref = await out.open_conversation(route.target, event.title)
                ids = await out.send(ref, event.body, attachments=event.attachments, actions=event.actions)
                deliveries.append(Delivery(route.channel, ref, list(ids or [])))
            except Exception as exc:  # noqa: BLE001 — one channel must not break the others
                logger.exception("Delivery of %s/%s to %s failed", event.type, event.severity, route.channel)
                deliveries.append(Delivery(route.channel, ref, error=str(exc) or exc.__class__.__name__))
        return deliveries

    async def open_conversation(self, event: Event) -> ConversationRef | None:
        """Open a conversation (topic) on the event's primary channel."""
        for route in self.policy.route(event):
            out = self._outbounds.get(route.channel)
            if out is None:
                continue
            try:
                return await out.open_conversation(route.target, event.title, owned=True)
            except Exception:  # noqa: BLE001
                logger.exception("open_conversation on %s failed for %s", route.channel, event.type)
        return None

    async def reply(
        self,
        ref: ConversationRef,
        text: str,
        *,
        actions: Sequence[Action] = (),
        attachments: Sequence[Attachment] = (),
        session_name: str | None = None,
    ) -> list[str]:
        out = self._outbounds.get(ref.channel)
        if out is None:
            logger.warning("reply: no outbound for channel %s", ref.channel)
            return []
        return list(await out.send(ref, text, attachments=attachments, actions=actions, session_name=session_name) or [])

    async def start_progress(self, ref: ConversationRef, **kwargs: Any) -> tuple[str, Any] | None:
        out = self._outbounds.get(ref.channel)
        if out is None:
            return None
        try:
            return ref.channel, await out.start_progress(ref, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("start_progress failed on %s", ref.channel)
            return None

    async def stop_progress(self, handle: tuple[str, Any] | None, *, ok: bool = True) -> None:
        if not handle:
            return
        channel, inner = handle
        out = self._outbounds.get(channel)
        if out is None:
            return
        try:
            await out.stop_progress(inner, ok=ok)
        except Exception:  # noqa: BLE001
            logger.exception("stop_progress failed on %s", channel)
