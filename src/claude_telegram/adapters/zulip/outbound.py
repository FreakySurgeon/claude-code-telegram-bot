"""Zulip OutboundChannel: stream/DM messages, reactions as progress, topic rename.

Conversation refs:
- stream: ``ConversationRef("zulip", "stream:<stream>", topic=<topic>)``
- DM:     ``ConversationRef("zulip", "dm:<email1>,<email2>")`` (emails sorted)

Progress is shown with reactions on the inbound message (👀 while working,
then ✅ or ❌) rather than a status message: Zulip notifies on every new
message, not on reactions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from ...markdown import extract_button_labels, markdown_to_zulip, split_text
from ...ports import Action, AckStatus, Attachment, Capabilities, ConversationRef
from .client import ZulipClient

logger = logging.getLogger(__name__)

MAX_LEN = 10000
TOPIC_MAX = 60
DEFAULT_TOPIC = "Divers"
DEFAULT_TOPIC_NAMES = ("", "(no topic)", "general chat", "général")

_ACK_EMOJI = {"done": "check", "error": "cross_mark"}


class OwnedTopics:
    """Topics opened by the agent: the inbound adapter answers there without a mention.

    Persisted under the ``owned`` key of the Zulip state file (shared with
    ZulipInbound, hence read-merge-write), bounded to the ``limit`` latest keys.
    """

    def __init__(self, path: Path, limit: int = 500):
        self.path = Path(path)
        self.limit = limit
        self._keys: list[str] = list(self._read().get("owned") or [])[-limit:]

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def add(self, key: str) -> None:
        if key in self._keys:
            self._keys.remove(key)
        self._keys.append(key)
        self._keys = self._keys[-self.limit:]
        data = self._read()
        data["owned"] = self._keys
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def __contains__(self, key: object) -> bool:
        return key in self._keys


def _stream_of(ref: ConversationRef) -> str | None:
    cid = ref.conversation_id
    return cid[len("stream:"):] if cid.startswith("stream:") else None


def _dm_recipients(ref: ConversationRef) -> list[str]:
    return [e for e in ref.conversation_id[len("dm:"):].split(",") if e]


class ZulipOutbound:
    name = "zulip"

    def __init__(
        self,
        client: ZulipClient,
        *,
        owned: OwnedTopics | None = None,
        default_topic_names: Sequence[str] = DEFAULT_TOPIC_NAMES,
    ):
        self.client = client
        self.owned = owned
        self.default_topic_names = {n.lower() for n in default_topic_names}

    def capabilities(self) -> Capabilities:
        return Capabilities(max_len=MAX_LEN, supports_edit=True, supports_topics=True)

    async def open_conversation(self, target: dict, title: str = "", *, owned: bool = False) -> ConversationRef:
        if target.get("dm"):
            ref = ConversationRef("zulip", "dm:" + ",".join(sorted(target["dm"])))
        else:
            stream = target.get("stream")
            if not stream:
                raise ValueError(f"Zulip target without stream or dm: {target!r}")
            topic = (target.get("topic") or title or DEFAULT_TOPIC)[:TOPIC_MAX]
            ref = ConversationRef("zulip", f"stream:{stream}", topic=topic)
        if owned and self.owned is not None:
            self.owned.add(ref.key)
        return ref

    async def _post(self, ref: ConversationRef, content: str) -> int:
        stream = _stream_of(ref)
        if stream is not None:
            return await self.client.send_stream(stream, ref.topic or DEFAULT_TOPIC, content)
        return await self.client.send_dm(_dm_recipients(ref), content)

    async def send(
        self,
        ref: ConversationRef,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
        actions: Sequence[Action] = (),
        session_name: str | None = None,
    ) -> list[str]:
        text, labels = extract_button_labels(text or "")
        labels += [a.label for a in actions]
        content = markdown_to_zulip(text)
        if labels:
            content += "\n\n_Options : " + " · ".join(labels) + "_"
        links = []
        for att in attachments:
            if att.path:
                path = Path(att.path)
                uri = await self.client.upload(path)
                links.append(f"[{att.name or path.name}]({uri})")
            elif att.url:
                links.append(f"[{att.name or att.url}]({att.url})")
        if links:
            content = (content + "\n\n" + "\n".join(links)).strip()
        if not content:
            return []
        return [str(await self._post(ref, chunk)) for chunk in split_text(content, MAX_LEN)]

    async def ack(self, ref: ConversationRef, message_id: str, status: AckStatus) -> None:
        mid = int(message_id)
        if status == "working":
            await self.client.add_reaction(mid, "eyes")
            return
        await self.client.remove_reaction(mid, "eyes")
        await self.client.add_reaction(mid, _ACK_EMOJI[status])

    async def edit(self, ref: ConversationRef, message_id: str, text: str) -> None:
        await self.client.update_message(int(message_id), content=markdown_to_zulip(text))

    async def delete(self, ref: ConversationRef, message_id: str) -> None:
        await self.client.delete_message(int(message_id))

    async def start_progress(
        self, ref: ConversationRef, *, inbound_message_id: str | None = None, **_: Any
    ) -> tuple[ConversationRef, str] | None:
        if not inbound_message_id:
            return None
        await self.ack(ref, inbound_message_id, "working")
        return ref, inbound_message_id

    async def stop_progress(self, handle: Any, *, ok: bool = True) -> None:
        if not handle:
            return
        ref, message_id = handle
        await self.ack(ref, message_id, "done" if ok else "error")

    async def rename_conversation(self, ref: ConversationRef, title: str) -> ConversationRef | None:
        stream = _stream_of(ref)
        title = (title or "").strip()[:TOPIC_MAX]
        if stream is None or not title or (ref.topic or "").lower() not in self.default_topic_names:
            return None
        narrow = [{"operator": "stream", "operand": stream}, {"operator": "topic", "operand": ref.topic or ""}]
        messages = await self.client.get_messages(narrow=narrow, anchor="newest", num_before=1)
        if not messages:
            return None
        await self.client.update_message(messages[-1]["id"], topic=title, propagate_mode="change_all")
        new_ref = ref.with_topic(title)
        if self.owned is not None and ref.key in self.owned:
            self.owned.add(new_ref.key)
        return new_ref
