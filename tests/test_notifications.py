"""Tests for NotificationService (publish / open_conversation / reply)."""

import pytest

from claude_telegram.notifications import NotificationService
from claude_telegram.ports import Action, Capabilities, ConversationRef, Event
from claude_telegram.routing import RoutingPolicy


class FakeOutbound:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.opened = []
        self.sent = []
        self.progress = []

    def capabilities(self):
        return Capabilities(max_len=1000, supports_edit=True, supports_topics=True)

    async def open_conversation(self, target, title="", *, owned=False):
        self.opened.append((target, title, owned))
        return ConversationRef(self.name, "c1", topic=target.get("topic") or title)

    async def send(self, ref, text, *, attachments=(), actions=(), session_name=None):
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append((ref, text, list(actions)))
        return ["m1"]

    async def start_progress(self, ref, **kw):
        self.progress.append(("start", ref, kw))
        return "h"

    async def stop_progress(self, handle, *, ok=True):
        self.progress.append(("stop", handle, ok))


CFG = {
    "default": "zulip",
    "urgent": ["telegram", "zulip"],
    "zulip": {"outputs": {"default": {"stream": "s", "topic": "Divers"}}},
    "telegram": {"only_severity": ["urgent"]},
}


@pytest.fixture
def outs():
    return {"zulip": FakeOutbound("zulip"), "telegram": FakeOutbound("telegram")}


async def test_publish_normal_goes_to_default_only(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    deliveries = await svc.publish(Event("x", "normal", body="hello"))
    assert [d.channel for d in deliveries] == ["zulip"]
    assert outs["zulip"].opened == [({"stream": "s", "topic": "Divers"}, "", False)]
    assert outs["zulip"].sent[0][1] == "hello"
    assert outs["telegram"].sent == []


async def test_publish_urgent_goes_to_both(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    deliveries = await svc.publish(Event("x", "urgent", body="fire"))
    assert [d.channel for d in deliveries] == ["telegram", "zulip"]
    assert all(d.error is None for d in deliveries)


async def test_failure_is_isolated(outs):
    outs["telegram"].fail = True
    svc = NotificationService(RoutingPolicy(CFG), outs)
    deliveries = await svc.publish(Event("x", "urgent", body="fire"))
    assert deliveries[0].error and "boom" in deliveries[0].error
    assert deliveries[1].error is None
    assert outs["zulip"].sent


async def test_unknown_channel_is_reported_not_raised():
    svc = NotificationService(RoutingPolicy(CFG), {})
    deliveries = await svc.publish(Event("x", "normal", body="b"))
    assert deliveries[0].channel == "zulip" and deliveries[0].error


async def test_conversation_hint_skips_open(outs):
    svc = NotificationService(RoutingPolicy.default(), outs)
    hint = ConversationRef("telegram", "42", thread_id=9)
    await svc.publish(Event("x", "normal", body="b", conversation_hint=hint))
    assert outs["telegram"].opened == []
    assert outs["telegram"].sent[0][0] == hint


async def test_actions_forwarded(outs):
    svc = NotificationService(RoutingPolicy.default(), outs)
    await svc.publish(Event("x", body="b", actions=[Action("Go", "go:1")]))
    assert outs["telegram"].sent[0][2] == [Action("Go", "go:1")]


async def test_open_conversation_uses_first_route_and_owned(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    ref = await svc.open_conversation(Event("fitness", "normal", title="Séance"))
    assert ref.channel == "zulip"
    assert outs["zulip"].opened[-1][2] is True


async def test_open_conversation_none_when_no_route():
    svc = NotificationService(RoutingPolicy(dict(CFG, default="telegram")), {"telegram": FakeOutbound("telegram")})
    assert await svc.open_conversation(Event("x", "normal")) is None


async def test_reply_routes_by_ref_channel(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    ref = ConversationRef("telegram", "42", thread_id=1)
    assert await svc.reply(ref, "hi") == ["m1"]
    assert outs["telegram"].sent[0][0] == ref


async def test_progress_roundtrip(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    ref = ConversationRef("zulip", "c", topic="t")
    handle = await svc.start_progress(ref, inbound_message_id="5")
    await svc.stop_progress(handle, ok=False)
    assert outs["zulip"].progress == [("start", ref, {"inbound_message_id": "5"}), ("stop", "h", False)]


MIRROR_CFG = dict(CFG, mirror={"briefing_morning": ["telegram"]})
BRIEFING = "Détail complet\n\n<!-- summary -->\n☀️ 3 priorités\n<!-- /summary -->"


async def test_mirror_sends_summary_to_telegram_and_full_body_to_zulip(outs):
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    deliveries = await svc.publish(Event("briefing_morning", "normal", title="Cron: morning", body=BRIEFING))
    assert [d.channel for d in deliveries] == ["zulip", "telegram"]
    assert outs["zulip"].sent[0][1] == "Détail complet"
    assert outs["telegram"].sent[0][1] == "☀️ 3 priorités"


async def test_mirror_opens_the_chat_root_never_a_topic(outs):
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    await svc.publish(Event("briefing_morning", "normal", title="Cron: morning", body=BRIEFING))
    assert outs["telegram"].opened == [({"bot": "gtd"}, "", False)]


async def test_mirror_truncates_when_there_is_no_summary_block(outs):
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    await svc.publish(Event("briefing_morning", "normal", body="\n".join(f"ligne {i}" for i in range(2000))))
    short = outs["telegram"].sent[0][1]
    assert 0 < len(short) <= 1200 and short.endswith("…")
    assert len(outs["zulip"].sent[0][1]) > 1200


async def test_mirror_still_reaches_telegram_when_the_primary_channel_is_down(outs):
    outs["zulip"].fail = True
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    deliveries = await svc.publish(Event("briefing_morning", "normal", body=BRIEFING))
    assert deliveries[0].error and deliveries[1].error is None
    assert outs["telegram"].sent[0][1] == "☀️ 3 priorités"


async def test_mirror_ignores_a_conversation_hint_of_the_same_channel(outs):
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    hint = ConversationRef("telegram", "42", thread_id=9)
    await svc.publish(Event("briefing_morning", "normal", body=BRIEFING, conversation_hint=hint))
    assert outs["telegram"].sent[0][0] != hint
    assert outs["telegram"].opened == [({"bot": "gtd"}, "", False)]


async def test_summary_block_is_stripped_even_without_a_mirror(outs):
    svc = NotificationService(RoutingPolicy(CFG), outs)
    await svc.publish(Event("briefing_morning", "normal", body=BRIEFING))
    assert outs["zulip"].sent[0][1] == "Détail complet"
    assert outs["telegram"].sent == []


async def test_non_mirrored_events_keep_their_body_verbatim(outs):
    svc = NotificationService(RoutingPolicy(MIRROR_CFG), outs)
    await svc.publish(Event("whatsapp_triage", "normal", body="  corps brut  \n"))
    assert outs["zulip"].sent[0][1] == "  corps brut  \n"
