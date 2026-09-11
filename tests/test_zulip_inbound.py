"""Tests for ZulipInbound (event-queue long-poll)."""

import json
import logging

import httpx
import pytest
import respx

from claude_telegram.adapters.zulip.client import ZulipClient
from claude_telegram.adapters.zulip.inbound import ZulipInbound
from claude_telegram.adapters.zulip.outbound import OwnedTopics
from claude_telegram.ports import Attachment

SITE = "https://zulip.example.com"
BOT = "agent-bot@example.com"
ME = {"result": "success", "user_id": 10, "email": BOT, "full_name": "Agent"}


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "zulip-events.json"


@pytest.fixture
def inbound(state_path):
    client = ZulipClient(SITE, BOT, "k")
    zi = ZulipInbound(
        client,
        state_path=state_path,
        listen_streams=["quotidien"],
        mention_streams=["maison"],
        dm=True,
        owned=OwnedTopics(state_path),
        bot_email=BOT,
    )
    zi.bot_name = "Agent"
    return zi


def stream_event(msg_id=1, stream="quotidien", topic="courses", content="salut", sender="thomas@example.com",
                 flags=(), event_id=0):
    return {
        "type": "message",
        "id": event_id,
        "flags": list(flags),
        "message": {
            "id": msg_id,
            "type": "stream",
            "display_recipient": stream,
            "subject": topic,
            "content": content,
            "sender_email": sender,
            "sender_full_name": "Thomas",
            "sender_id": 8,
        },
    }


def dm_event(msg_id=2, content="hello"):
    return {
        "type": "message",
        "id": 0,
        "flags": [],
        "message": {
            "id": msg_id,
            "type": "private",
            "display_recipient": [
                {"email": "thomas@example.com", "full_name": "Thomas", "id": 8},
                {"email": BOT, "full_name": "Agent", "id": 10},
            ],
            "subject": "",
            "content": content,
            "sender_email": "thomas@example.com",
            "sender_full_name": "Thomas",
            "sender_id": 8,
        },
    }


def test_own_message_ignored_and_logged(inbound, caplog):
    with caplog.at_level(logging.INFO):
        assert inbound.to_inbound(stream_event(sender=BOT)) is None
    assert "ignored own message" in caplog.text


def test_listen_stream_without_mention(inbound):
    msg = inbound.to_inbound(stream_event(content="combien de cartes ?"))
    assert msg.text == "combien de cartes ?"
    assert msg.reply_to.conversation_id == "stream:quotidien"
    assert msg.reply_to.topic == "courses"
    assert msg.message_id == "1" and msg.user_name == "Thomas" and msg.user == "thomas@example.com"


def test_mention_stream_requires_mention(inbound):
    assert inbound.to_inbound(stream_event(stream="maison", content="bla")) is None
    msg = inbound.to_inbound(
        stream_event(stream="maison", content="@**Agent** tu peux noter ?", flags=["mentioned"])
    )
    assert msg.text == "tu peux noter ?"
    assert msg.is_mention


def test_unknown_stream_ignored(inbound):
    assert inbound.to_inbound(stream_event(stream="autre", flags=["mentioned"])) is None


def test_non_message_event_ignored(inbound):
    assert inbound.to_inbound({"type": "heartbeat", "id": 3}) is None


def test_dm(inbound):
    msg = inbound.to_inbound(dm_event())
    assert msg.conversation_id == "dm:thomas@example.com"
    assert msg.reply_to.topic is None


def test_dm_disabled(inbound):
    inbound.dm = False
    assert inbound.to_inbound(dm_event()) is None


def test_owned_topic_without_mention(inbound):
    inbound.owned.add("zulip:stream:maison:Séance 2026-09-11")
    msg = inbound.to_inbound(stream_event(stream="maison", topic="Séance 2026-09-11", content="fait"))
    assert msg is not None and msg.text == "fait"


def test_upload_link_becomes_attachment(inbound):
    msg = inbound.to_inbound(stream_event(content="regarde [photo.jpg](/user_uploads/2/ab/photo.jpg)"))
    assert msg.attachments == [Attachment("image", url="/user_uploads/2/ab/photo.jpg", name="photo.jpg")]
    msg = inbound.to_inbound(stream_event(msg_id=5, content="[vocal.m4a](/user_uploads/2/cd/vocal.m4a)"))
    assert msg.attachments[0].kind == "audio"


def test_seen_message_ignored_and_persisted(inbound, state_path):
    inbound.mark_seen(1)
    inbound.save_state()
    assert inbound.seen("1")
    assert inbound.to_inbound(stream_event(msg_id=1)) is None
    assert 1 in json.loads(state_path.read_text())["seen"]


def _mock_common(events_side_effect):
    respx.get(f"{SITE}/api/v1/users/me").mock(return_value=httpx.Response(200, json=ME))
    register = respx.post(f"{SITE}/api/v1/register").mock(
        return_value=httpx.Response(200, json={"result": "success", "queue_id": "q1", "last_event_id": -1})
    )
    events = respx.get(f"{SITE}/api/v1/events").mock(side_effect=events_side_effect)
    return register, events


def _ok_events(*evs):
    return httpx.Response(200, json={"result": "success", "events": list(evs)})


@respx.mock
async def test_run_registers_dispatches_and_persists(inbound, state_path):
    register, _ = _mock_common([_ok_events(stream_event(msg_id=7, event_id=0))])
    got = []

    async def handler(msg):
        got.append(msg)
        inbound.stop()

    await inbound.run(handler)
    assert register.call_count == 1
    assert [m.message_id for m in got] == ["7"]
    state = json.loads(state_path.read_text())
    assert state["queue_id"] == "q1" and state["last_event_id"] == 0 and 7 in state["seen"]


@respx.mock
async def test_run_reregisters_on_bad_event_queue_id(inbound, state_path):
    state_path.write_text(json.dumps({"queue_id": "old", "last_event_id": 4}))
    inbound.load_state()
    bad = httpx.Response(400, json={"result": "error", "code": "BAD_EVENT_QUEUE_ID", "msg": "Bad event queue id: old"})
    register, events = _mock_common([bad, _ok_events(stream_event(msg_id=8, event_id=0))])

    async def handler(msg):
        inbound.stop()

    await inbound.run(handler)
    assert register.call_count == 1
    assert events.calls[0].request.url.params["queue_id"] == "old"
    assert events.calls[1].request.url.params["queue_id"] == "q1"


@respx.mock
async def test_run_resumes_existing_queue_without_register(inbound, state_path):
    state_path.write_text(json.dumps({"queue_id": "q9", "last_event_id": 12}))
    inbound.load_state()
    register, events = _mock_common([_ok_events(stream_event(msg_id=9, event_id=13))])

    async def handler(msg):
        inbound.stop()

    await inbound.run(handler)
    assert register.call_count == 0
    params = events.calls[0].request.url.params
    assert params["queue_id"] == "q9" and params["last_event_id"] == "12"
    assert json.loads(state_path.read_text())["last_event_id"] == 13


@respx.mock
async def test_fetch_attachment_writes_temp_file(inbound):
    respx.get(f"{SITE}/api/v1/user_uploads/2/ab/photo.jpg").mock(
        return_value=httpx.Response(200, content=b"JPEG", headers={"content-type": "image/jpeg"})
    )
    path = await inbound.fetch_attachment(Attachment("image", url="/user_uploads/2/ab/photo.jpg", name="photo.jpg"))
    assert path.suffix == ".jpg" and path.read_bytes() == b"JPEG"
    path.unlink()
