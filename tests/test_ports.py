"""Tests for channel-independent ports (data types)."""

from claude_telegram.ports import Attachment, ConversationRef, Event, InboundMessage


def test_ref_key_topic_and_thread():
    assert ConversationRef("zulip", "stream:quotidien", topic="Courses").key == "zulip:stream:quotidien:Courses"
    assert ConversationRef("telegram", "42", thread_id=7).key == "telegram:42:7"
    assert ConversationRef("telegram", "42").key == "telegram:42:"


def test_ref_roundtrip_dict():
    ref = ConversationRef("telegram", "42", thread_id=7, bot="gtd")
    assert ConversationRef.from_dict(ref.to_dict()) == ref


def test_ref_from_dict_ignores_unknown_keys():
    assert ConversationRef.from_dict({"channel": "zulip", "conversation_id": "x", "junk": 1}) == ConversationRef("zulip", "x")


def test_ref_with_topic():
    ref = ConversationRef("zulip", "stream:q", topic="a")
    assert ref.with_topic("b") == ConversationRef("zulip", "stream:q", topic="b")


def test_event_defaults():
    e = Event(type="briefing_morning")
    assert e.severity == "info"
    assert e.attachments == [] and e.actions == [] and e.metadata == {}
    assert e.conversation_hint is None


def test_inbound_ref_defaults_to_conversation():
    m = InboundMessage(channel="zulip", conversation_id="dm:a@x", user="a@x", text="hi")
    assert m.ref == ConversationRef("zulip", "dm:a@x")


def test_inbound_ref_uses_reply_to():
    ref = ConversationRef("zulip", "stream:q", topic="t")
    m = InboundMessage(channel="zulip", conversation_id="stream:q", user="a", text="hi", reply_to=ref,
                       attachments=[Attachment("image", url="/u/x.png")])
    assert m.ref == ref
    assert m.attachments[0].kind == "image"
