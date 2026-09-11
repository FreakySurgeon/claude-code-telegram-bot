"""Channel-independent ports (hexagonal architecture).

The core (queue, crons, webhooks, notifications) only knows these types.
Everything that talks to a chat API lives under ``adapters/<name>/`` and
implements ``InboundChannel`` and/or ``OutboundChannel``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence

from .claude import ClaudeResult

Severity = Literal["urgent", "normal", "info"]
AckStatus = Literal["working", "done", "error"]

# What an AgentRunner returns (ClaudeRunner today, possibly behind a provider chain).
AgentResult = ClaudeResult


@dataclass(frozen=True)
class Attachment:
    kind: str  # "image" | "audio" | "file"
    url: str | None = None
    path: str | None = None
    mime: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ConversationRef:
    """Where a conversation lives on a channel (chat + thread/topic)."""

    channel: str
    conversation_id: str
    thread_id: int | None = None
    topic: str | None = None
    bot: str | None = None

    @property
    def key(self) -> str:
        sub = self.thread_id if self.thread_id is not None else (self.topic or "")
        return f"{self.channel}:{self.conversation_id}:{sub}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationRef":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def with_topic(self, topic: str) -> "ConversationRef":
        return replace(self, topic=topic)


@dataclass
class InboundMessage:
    channel: str
    conversation_id: str
    user: str
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    is_mention: bool = False
    reply_to: ConversationRef | None = None
    message_id: str | None = None
    user_name: str = ""

    @property
    def ref(self) -> ConversationRef:
        return self.reply_to or ConversationRef(self.channel, self.conversation_id)


@dataclass(frozen=True)
class Capabilities:
    max_len: int
    supports_edit: bool
    supports_topics: bool


@dataclass(frozen=True)
class Action:
    """A quick-reply action (inline button on Telegram, plain text elsewhere)."""

    label: str
    data: str


@dataclass
class Event:
    """Something the core wants to tell the user; routed by RoutingPolicy."""

    type: str
    severity: Severity = "info"
    title: str = ""
    body: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    conversation_hint: ConversationRef | None = None
    actions: list[Action] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


InboundHandler = Callable[[InboundMessage], Awaitable[None]]


class InboundChannel(Protocol):
    name: str

    async def run(self, handler: InboundHandler) -> None: ...

    async def fetch_attachment(self, att: Attachment) -> Path: ...


class OutboundChannel(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    async def open_conversation(self, target: dict, title: str = "", *, owned: bool = False) -> ConversationRef: ...

    async def send(
        self,
        ref: ConversationRef,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
        actions: Sequence[Action] = (),
        session_name: str | None = None,
    ) -> list[str]: ...

    async def ack(self, ref: ConversationRef, message_id: str, status: AckStatus) -> None: ...

    async def edit(self, ref: ConversationRef, message_id: str, text: str) -> None: ...

    async def delete(self, ref: ConversationRef, message_id: str) -> None: ...

    async def start_progress(
        self,
        ref: ConversationRef,
        *,
        inbound_message_id: str | None = None,
        continue_session: bool = False,
        session_name: str | None = None,
    ) -> Any: ...

    async def stop_progress(self, handle: Any, *, ok: bool = True) -> None: ...

    async def rename_conversation(self, ref: ConversationRef, title: str) -> ConversationRef | None: ...


class AgentRunner(Protocol):
    async def run(self, message: str, **kwargs: Any) -> AgentResult: ...
