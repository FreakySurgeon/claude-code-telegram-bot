import json
from pathlib import Path

import httpx
import pytest
import respx

from claude_telegram.adapters.zulip.client import ZulipClient, ZulipError

SITE = "https://zulip.example.com"


@pytest.fixture
def client():
    c = ZulipClient(SITE, "bot@example.com", "fake-api-key")
    yield c


@pytest.fixture
def client_with_host():
    c = ZulipClient(
        SITE,
        "bot@example.com",
        "fake-api-key",
        host_header="zulip.example.com",
    )
    yield c


@respx.mock
async def test_send_stream_posts_expected_form_fields(client):
    route = respx.post(f"{SITE}/api/v1/messages").mock(
        return_value=httpx.Response(200, json={"result": "success", "id": 42})
    )

    message_id = await client.send_stream("general", "hello", "hi there")

    assert message_id == 42
    request = route.calls.last.request
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["type"] == "stream"
    assert form["to"] == "general"
    assert form["topic"] == "hello"
    assert form["content"] == "hi there"


@respx.mock
async def test_host_header_sent_when_configured(client_with_host):
    route = respx.post(f"{SITE}/api/v1/messages").mock(
        return_value=httpx.Response(200, json={"result": "success", "id": 1})
    )

    await client_with_host.send_stream("general", "hello", "hi there")

    assert route.calls.last.request.headers["host"] == "zulip.example.com"


@respx.mock
async def test_send_dm_encodes_to_as_json_list(client):
    route = respx.post(f"{SITE}/api/v1/messages").mock(
        return_value=httpx.Response(200, json={"result": "success", "id": 7})
    )

    message_id = await client.send_dm(["a@example.com", "b@example.com"], "hi")

    assert message_id == 7
    request = route.calls.last.request
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["type"] == "direct"
    assert json.loads(form["to"]) == ["a@example.com", "b@example.com"]


@respx.mock
async def test_get_events_bad_queue_id_raises_zulip_error(client):
    respx.get(f"{SITE}/api/v1/events").mock(
        return_value=httpx.Response(
            400,
            json={
                "result": "error",
                "code": "BAD_EVENT_QUEUE_ID",
                "msg": "Bad event queue id",
            },
        )
    )

    with pytest.raises(ZulipError) as exc_info:
        await client.get_events("queue-1", 0)

    assert exc_info.value.code == "BAD_EVENT_QUEUE_ID"


@respx.mock
async def test_register_returns_queue_id_and_last_event_id(client):
    respx.post(f"{SITE}/api/v1/register").mock(
        return_value=httpx.Response(
            200, json={"result": "success", "queue_id": "q-1", "last_event_id": -1}
        )
    )

    queue_id, last_event_id = await client.register()

    assert (queue_id, last_event_id) == ("q-1", -1)


@respx.mock
async def test_add_reaction_hits_expected_route(client):
    route = respx.post(f"{SITE}/api/v1/messages/99/reactions").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )

    await client.add_reaction(99, "eyes")

    request = route.calls.last.request
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["emoji_name"] == "eyes"


@respx.mock
async def test_remove_reaction_swallows_error(client):
    respx.delete(f"{SITE}/api/v1/messages/99/reactions").mock(
        return_value=httpx.Response(
            400, json={"result": "error", "code": "REACTION_DOES_NOT_EXIST", "msg": "nope"}
        )
    )

    # Should not raise.
    await client.remove_reaction(99, "eyes")


@respx.mock
async def test_update_message_with_topic_sends_propagate_mode(client):
    route = respx.patch(f"{SITE}/api/v1/messages/5").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )

    await client.update_message(5, topic="New topic")

    request = route.calls.last.request
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["topic"] == "New topic"
    assert form["propagate_mode"] == "change_all"
    assert "content" not in form


@respx.mock
async def test_upload_returns_uri(client, tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello")
    respx.post(f"{SITE}/api/v1/user_uploads").mock(
        return_value=httpx.Response(
            200, json={"result": "success", "uri": "/user_uploads/2/ab/note.txt"}
        )
    )

    uri = await client.upload(file_path)

    assert uri == "/user_uploads/2/ab/note.txt"


@respx.mock
async def test_download_follows_temporary_url(client):
    respx.get(f"{SITE}/api/v1/user_uploads/2/ab/note.txt").mock(
        return_value=httpx.Response(
            200, json={"result": "success", "url": "/user_uploads/temp/xyz/note.txt"}
        )
    )
    respx.get(f"{SITE}/user_uploads/temp/xyz/note.txt").mock(
        return_value=httpx.Response(200, content=b"file-bytes")
    )

    content = await client.download("/user_uploads/2/ab/note.txt")

    assert content == b"file-bytes"
