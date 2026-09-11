"""Minimal async HTTP client for the Zulip REST API.

Wraps the subset of the Zulip API used by the Zulip channel adapter:
sending messages, reactions, message edits/deletes, file uploads/downloads
and the events (long-poll) queue.

The server is typically reached over a LAN IP while the public hostname is
still expected in the TLS SNI / ``Host`` header by the reverse proxy in
front of it; ``host_header`` lets callers override the ``Host`` header
independently of ``site`` for that case.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ZulipError(Exception):
    """Raised when the Zulip API returns ``result != "success"``."""

    def __init__(self, msg: str, code: str | None = None) -> None:
        super().__init__(msg)
        self.msg = msg
        self.code = code


class ZulipClient:
    def __init__(
        self,
        site: str,
        email: str,
        api_key: str,
        host_header: str | None = None,
        timeout: float = 30,
    ) -> None:
        self.site = site.rstrip("/")
        self.email = email
        self.api_key = api_key
        self.host_header = host_header
        self.timeout = timeout

        headers = {}
        if host_header:
            headers["Host"] = host_header

        self._client = httpx.AsyncClient(
            base_url=self.site,
            auth=(email, api_key),
            headers=headers,
            timeout=timeout,
        )

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if data.get("result") != "success":
            raise ZulipError(data.get("msg", "Zulip API error"), data.get("code"))
        return data

    async def get_me(self) -> dict:
        response = await self._client.get("/api/v1/users/me")
        return self._check(response)

    async def send_stream(self, stream: str, topic: str, content: str) -> int:
        response = await self._client.post(
            "/api/v1/messages",
            data={
                "type": "stream",
                "to": stream,
                "topic": topic,
                "content": content,
            },
        )
        data = self._check(response)
        return data["id"]

    async def send_dm(self, to: list[str], content: str) -> int:
        response = await self._client.post(
            "/api/v1/messages",
            data={
                "type": "direct",
                "to": json.dumps(to),
                "content": content,
            },
        )
        data = self._check(response)
        return data["id"]

    async def add_reaction(self, message_id: int, emoji: str) -> None:
        response = await self._client.post(
            f"/api/v1/messages/{message_id}/reactions",
            data={"emoji_name": emoji},
        )
        self._check(response)

    async def remove_reaction(self, message_id: int, emoji: str) -> None:
        try:
            response = await self._client.request(
                "DELETE",
                f"/api/v1/messages/{message_id}/reactions",
                params={"emoji_name": emoji},
            )
            self._check(response)
        except ZulipError as exc:
            logger.debug("remove_reaction ignored error: %s", exc)

    async def update_message(
        self,
        message_id: int,
        *,
        content: str | None = None,
        topic: str | None = None,
        propagate_mode: str = "change_all",
    ) -> None:
        data: dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if topic is not None:
            data["topic"] = topic
            data["propagate_mode"] = propagate_mode
        response = await self._client.patch(f"/api/v1/messages/{message_id}", data=data)
        self._check(response)

    async def delete_message(self, message_id: int) -> None:
        response = await self._client.request("DELETE", f"/api/v1/messages/{message_id}")
        self._check(response)

    async def get_messages(
        self,
        *,
        narrow: list[dict],
        anchor: str = "newest",
        num_before: int = 1,
        num_after: int = 0,
    ) -> list[dict]:
        response = await self._client.get(
            "/api/v1/messages",
            params={
                "narrow": json.dumps(narrow),
                "anchor": anchor,
                "num_before": num_before,
                "num_after": num_after,
                "apply_markdown": "false",
            },
        )
        data = self._check(response)
        return data["messages"]

    async def upload(self, path: Path) -> str:
        with open(path, "rb") as fh:
            response = await self._client.post(
                "/api/v1/user_uploads",
                files={"file": (path.name, fh)},
            )
        data = self._check(response)
        return data.get("uri") or data.get("url")

    async def download(self, upload_path: str) -> bytes:
        response = await self._client.get(f"/api/v1{upload_path}")
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type:
            response.raise_for_status()
            return response.content
        data = self._check(response)
        temp_url = data["url"]
        temp_response = await self._client.get(temp_url)
        temp_response.raise_for_status()
        return temp_response.content

    async def register(self, event_types: tuple[str, ...] = ("message",)) -> tuple[str, int]:
        response = await self._client.post(
            "/api/v1/register",
            data={
                "event_types": json.dumps(list(event_types)),
                "apply_markdown": "false",
                "client_gravatar": "false",
            },
        )
        data = self._check(response)
        return data["queue_id"], data["last_event_id"]

    async def get_events(
        self, queue_id: str, last_event_id: int, timeout: float = 90
    ) -> list[dict]:
        response = await self._client.get(
            "/api/v1/events",
            params={"queue_id": queue_id, "last_event_id": last_event_id},
            timeout=timeout + 10,
        )
        data = self._check(response)
        return data["events"]

    async def aclose(self) -> None:
        await self._client.aclose()
