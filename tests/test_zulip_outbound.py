"""Tests for ZulipOutbound (respx-mocked Zulip API)."""

import json

import httpx
import pytest
import respx

from claude_telegram.adapters.zulip.client import ZulipClient
from claude_telegram.adapters.zulip.outbound import OwnedTopics, ZulipOutbound
from claude_telegram.ports import Action, Attachment, ConversationRef

SITE = "https://zulip.example.com"


def _form(request) -> dict:
    return dict(httpx.QueryParams(request.content.decode()))


@pytest.fixture
def out(tmp_path):
    client = ZulipClient(SITE, "bot@example.com", "k")
    return ZulipOutbound(client, owned=OwnedTopics(tmp_path / "zulip-events.json"))


def _mock_send(start_id=100):
    ids = iter(range(start_id, start_id + 50))
    return respx.post(f"{SITE}/api/v1/messages").mock(
        side_effect=lambda req: httpx.Response(200, json={"result": "success", "id": next(ids)})
    )


async def test_open_conversation_stream_and_dm(out):
    ref = await out.open_conversation({"stream": "agent", "topic": "Infra"})
    assert ref == ConversationRef("zulip", "stream:agent", topic="Infra")
    ref = await out.open_conversation({"stream": "agent"}, "Mon titre")
    assert ref.topic == "Mon titre"
    ref = await out.open_conversation({"stream": "agent"})
    assert ref.topic == "Divers"
    ref = await out.open_conversation({"dm": ["b@x", "a@x"]})
    assert ref.conversation_id == "dm:a@x,b@x"


async def test_open_conversation_owned_persists_key(out, tmp_path):
    ref = await out.open_conversation({"stream": "sport", "topic": "Séance"}, owned=True)
    assert ref.key in out.owned
    reloaded = OwnedTopics(tmp_path / "zulip-events.json")
    assert ref.key in reloaded


def test_owned_topics_keeps_other_keys_and_is_bounded(tmp_path):
    path = tmp_path / "zulip-events.json"
    path.write_text(json.dumps({"queue_id": "q1", "last_event_id": 5}))
    owned = OwnedTopics(path, limit=3)
    for i in range(5):
        owned.add(f"k{i}")
    data = json.loads(path.read_text())
    assert data["queue_id"] == "q1" and data["last_event_id"] == 5
    assert data["owned"] == ["k2", "k3", "k4"]
    assert "k0" not in owned and "k4" in owned


@respx.mock
async def test_send_stream_splits_in_two_chunks(out):
    route = _mock_send()
    ref = ConversationRef("zulip", "stream:agent", topic="Infra")
    ids = await out.send(ref, ("a" * 99 + "\n") * 150)  # 15 000 chars
    assert ids == ["100", "101"]
    assert route.call_count == 2
    form = _form(route.calls[0].request)
    assert form["type"] == "stream" and form["to"] == "agent" and form["topic"] == "Infra"
    assert all(len(_form(c.request)["content"]) <= 10000 for c in route.calls)


@respx.mock
async def test_send_dm(out):
    route = _mock_send()
    ref = ConversationRef("zulip", "dm:a@x,b@x")
    assert await out.send(ref, "salut") == ["100"]
    form = _form(route.calls.last.request)
    assert form["type"] == "direct"
    assert json.loads(form["to"]) == ["a@x", "b@x"]


@respx.mock
async def test_send_renders_buttons_as_text_and_strips_title(out):
    route = _mock_send()
    ref = ConversationRef("zulip", "stream:agent", topic="t")
    text = 'Choisis.\n<!-- title: Mon sujet -->\n<!-- buttons: ["Oui", "Non"] -->'
    await out.send(ref, text, actions=[Action("Plus tard", "later")])
    content = _form(route.calls.last.request)["content"]
    assert "title" not in content and "<!--" not in content
    assert "_Options : Oui · Non · Plus tard_" in content
    assert content.startswith("Choisis.")


@respx.mock
async def test_send_uploads_attachment_with_path(out, tmp_path):
    f = tmp_path / "rapport.pdf"
    f.write_bytes(b"%PDF")
    respx.post(f"{SITE}/api/v1/user_uploads").mock(
        return_value=httpx.Response(200, json={"result": "success", "uri": "/user_uploads/1/ab/rapport.pdf"})
    )
    route = _mock_send()
    ref = ConversationRef("zulip", "stream:agent", topic="t")
    await out.send(ref, "Voici", attachments=[Attachment("file", path=str(f), name="rapport.pdf")])
    content = _form(route.calls.last.request)["content"]
    assert "[rapport.pdf](/user_uploads/1/ab/rapport.pdf)" in content


@respx.mock
async def test_ack_done_swaps_eyes_for_check(out):
    add = respx.post(f"{SITE}/api/v1/messages/7/reactions").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )
    remove = respx.delete(f"{SITE}/api/v1/messages/7/reactions").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )
    ref = ConversationRef("zulip", "stream:agent", topic="t")
    await out.ack(ref, "7", "done")
    assert remove.calls.last.request.url.params["emoji_name"] == "eyes"
    assert _form(add.calls.last.request)["emoji_name"] == "check"


@respx.mock
async def test_progress_acks_working_then_error(out):
    add = respx.post(f"{SITE}/api/v1/messages/9/reactions").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )
    respx.delete(f"{SITE}/api/v1/messages/9/reactions").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )
    ref = ConversationRef("zulip", "stream:agent", topic="t")
    handle = await out.start_progress(ref, inbound_message_id="9")
    await out.stop_progress(handle, ok=False)
    emojis = [_form(c.request)["emoji_name"] for c in add.calls]
    assert emojis == ["eyes", "cross_mark"]
    assert await out.start_progress(ref) is None
    await out.stop_progress(None)


@respx.mock
async def test_rename_ignored_when_topic_not_default(out):
    route = respx.get(f"{SITE}/api/v1/messages")
    ref = ConversationRef("zulip", "stream:agent", topic="Sujet choisi")
    assert await out.rename_conversation(ref, "Nouveau") is None
    assert not route.called
    assert await out.rename_conversation(ConversationRef("zulip", "dm:a@x"), "Nouveau") is None


@respx.mock
async def test_rename_default_topic_patches_last_message(out):
    respx.get(f"{SITE}/api/v1/messages").mock(
        return_value=httpx.Response(200, json={"result": "success", "messages": [{"id": 55}]})
    )
    patch = respx.patch(f"{SITE}/api/v1/messages/55").mock(
        return_value=httpx.Response(200, json={"result": "success"})
    )
    ref = ConversationRef("zulip", "stream:quotidien", topic="(no topic)")
    new_ref = await out.rename_conversation(ref, "Courses du samedi")
    assert new_ref.topic == "Courses du samedi"
    form = _form(patch.calls.last.request)
    assert form["topic"] == "Courses du samedi" and form["propagate_mode"] == "change_all"


def test_build_outbounds_adds_zulip_only_when_configured(tmp_path):
    from claude_telegram.adapters import build_outbounds
    from claude_telegram.routing import RoutingPolicy

    assert set(build_outbounds(RoutingPolicy.default(), {}, tmp_path)) == {"telegram"}
    policy = RoutingPolicy({"default": "zulip", "zulip": {"site": SITE, "bot_email": "b@x", "api_key": "k"}})
    outs = build_outbounds(policy, {}, tmp_path)
    assert set(outs) == {"telegram", "zulip"}
    assert outs["zulip"].owned.path == tmp_path / "zulip-events.json"
