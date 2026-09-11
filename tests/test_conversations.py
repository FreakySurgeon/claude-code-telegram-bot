"""Tests for ConversationService / SessionStore / runner_key."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_telegram.conversations import ConversationService, SessionStore, runner_key
from claude_telegram.ports import Attachment, ConversationRef, InboundMessage

REF = ConversationRef("zulip", "stream:quotidien", topic="Courses")


class FakeRunner:
    def __init__(self):
        self.session_id = None
        self.is_running = False
        self.cancel = AsyncMock(return_value=True)


class FakeSessions:
    def __init__(self):
        self.runners = {}

    def get_session(self, working_dir=None, *, thread_id=0):
        return self.runners.setdefault((working_dir, thread_id), FakeRunner())


class FakeNotifications:
    def __init__(self):
        self.replies = []
        self.out = MagicMock()
        self.out.ack = AsyncMock()

    async def reply(self, ref, text, **kwargs):
        self.replies.append((ref, text))

    def outbound(self, channel):
        return self.out


@pytest.fixture
def env(tmp_path):
    submitted = []

    async def submit(item):
        submitted.append(item)
        return len(submitted)

    store = SessionStore(tmp_path / "channel-sessions.json")
    inbound = MagicMock()
    inbound.fetch_attachment = AsyncMock()
    transcribe = AsyncMock(return_value=SimpleNamespace(text="acheter du pain"))
    service = ConversationService(
        notifications=FakeNotifications(),
        queue_submit=submit,
        sessions_manager=FakeSessions(),
        working_dir="/work",
        session_store=store,
        data_dir=tmp_path,
        channel_contexts={"zulip": "Canal : Zulip"},
        inbound_channels={"zulip": inbound},
        transcribe=transcribe,
    )
    return SimpleNamespace(service=service, submitted=submitted, store=store, inbound=inbound,
                           transcribe=transcribe, tmp=tmp_path)


def msg(text="salut", attachments=(), message_id="42"):
    return InboundMessage(channel="zulip", conversation_id=REF.conversation_id, user="t@example.com",
                          text=text, attachments=list(attachments), reply_to=REF, message_id=message_id,
                          user_name="Thomas")


def _age_session(store, key, hours, session_id="sess-1"):
    data = {key: {"session_id": session_id,
                  "last_interaction": (datetime.now() - timedelta(hours=hours)).isoformat()}}
    store.path.write_text(json.dumps(data))


def test_runner_key_is_stable_negative_for_zulip_and_thread_for_telegram():
    assert runner_key(REF) == runner_key(ConversationRef("zulip", "stream:quotidien", topic="Courses"))
    assert runner_key(REF) < 0
    assert runner_key(REF) != runner_key(REF.with_topic("Autre"))
    assert runner_key(ConversationRef("telegram", "1", thread_id=7)) == 7
    assert runner_key(ConversationRef("telegram", "1")) == 0


def test_session_store_persists_and_moves(tmp_path):
    path = tmp_path / "s.json"
    SessionStore(path).save("a", "sess")
    store = SessionStore(path)
    assert store.get("a")["session_id"] == "sess"
    store.move("a", "b")
    assert store.get("a") is None and store.get("b")["session_id"] == "sess"


async def test_text_message_becomes_queue_item(env):
    await env.service.handle(msg("combien de cartes ?"))
    [item] = env.submitted
    assert item.prompt == "[zulip · #quotidien › Courses · de Thomas]\ncombien de cartes ?"
    assert item.conversation == REF and item.source == "zulip"
    assert item.channel_context == "Canal : Zulip"
    assert item.new_session and not item.continue_session
    assert item.metadata == {"inbound_message_id": "42", "type": "zulip-conversation"}


async def test_recent_session_continues(env):
    _age_session(env.store, REF.key, hours=1)
    await env.service.handle(msg())
    [item] = env.submitted
    assert item.continue_session and not item.new_session
    assert env.service.runner_for(REF).session_id == "sess-1"


async def test_stale_session_starts_new(env):
    _age_session(env.store, REF.key, hours=13)
    await env.service.handle(msg())
    [item] = env.submitted
    assert item.new_session and not item.continue_session
    assert env.service.sessions.get_session("/work", thread_id=runner_key(REF)).session_id is None


async def test_new_command_resets_session(env):
    _age_session(env.store, REF.key, hours=1)
    await env.service.handle(msg("/new"))
    assert env.submitted == []
    assert env.store.get(REF.key)["session_id"] is None
    assert env.service.notifications.replies == [(REF, "🆕 Nouvelle session.")]
    await env.service.handle(msg("et maintenant"))
    assert env.submitted[0].new_session


async def test_cancel_command(env):
    await env.service.handle(msg("/cancel"))
    env.service.runner_for(REF).cancel.assert_awaited_once()
    assert env.service.notifications.replies[-1][1] == "🛑 Annulé."


async def test_image_attachment_path_in_prompt(env, tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"JPEG")
    env.inbound.fetch_attachment.return_value = img
    await env.service.handle(msg("regarde", [Attachment("image", url="/user_uploads/1/photo.jpg", name="photo.jpg")]))
    assert f"[Image jointe : {img}]" in env.submitted[0].prompt


async def test_audio_attachment_is_transcribed(env, tmp_path):
    audio = tmp_path / "vocal.m4a"
    audio.write_bytes(b"AUDIO")
    env.inbound.fetch_attachment.return_value = audio
    await env.service.handle(msg("", [Attachment("audio", url="/user_uploads/1/vocal.m4a", name="vocal.m4a")]))
    env.transcribe.assert_awaited_once_with(str(audio))
    assert "[Vocal transcrit] acheter du pain" in env.submitted[0].prompt


async def test_file_attachment_copied_to_inbox(env, tmp_path):
    doc = tmp_path / "devis.pdf"
    doc.write_bytes(b"%PDF")
    env.inbound.fetch_attachment.return_value = doc
    await env.service.handle(msg("", [Attachment("file", url="/user_uploads/1/devis.pdf", name="devis.pdf")]))
    assert (env.tmp / "inbox" / "devis.pdf").read_bytes() == b"%PDF"
    assert "[Fichier reçu :" in env.submitted[0].prompt


async def test_queue_full_replies_and_acks_error(env):
    async def full(item):
        return None

    env.service.queue_submit = full
    await env.service.handle(msg())
    assert env.service.notifications.replies == [(REF, "⚠️ File pleine, réessaie dans quelques minutes.")]
    env.service.notifications.out.ack.assert_awaited_once_with(REF, "42", "error")


async def test_dm_place_label(env):
    dm = ConversationRef("zulip", "dm:t@example.com")
    await env.service.handle(InboundMessage(channel="zulip", conversation_id=dm.conversation_id, user="t@example.com",
                                            text="hello", reply_to=dm, message_id="1", user_name="Thomas"))
    assert env.submitted[0].prompt.startswith("[zulip · DM · de Thomas]\n")
