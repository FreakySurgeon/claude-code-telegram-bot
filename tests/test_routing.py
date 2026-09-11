"""Tests for RoutingPolicy (event type/severity -> channels)."""

import datetime as dt
import os
from pathlib import Path

from claude_telegram.ports import Event
from claude_telegram.routing import ChannelRef, RoutingPolicy, expand_env

CFG = {
    "default": "zulip",
    "urgent": ["telegram", "zulip"],
    "zulip": {
        "outputs": {
            "briefing_morning": {"stream": "s", "topic": "Briefing matin"},
            "fitness": {"stream": "sport", "topic": "Séance {date}"},
            "calendar_action": {"stream": "s", "topic": "{title}"},
            "default": {"stream": "s", "topic": "Divers"},
        }
    },
    "telegram": {"bot": "gtd", "only_severity": ["urgent"], "dev_bot_events": ["completed", "waiting"]},
}


def test_default_policy_everything_to_telegram():
    p = RoutingPolicy.default()
    for sev in ("urgent", "normal", "info"):
        assert p.route(Event("briefing_morning", sev)) == [ChannelRef("telegram", {"bot": "gtd"})]


def test_normal_goes_to_default_with_output_target():
    assert RoutingPolicy(CFG).route(Event("briefing_morning", "normal")) == [
        ChannelRef("zulip", {"stream": "s", "topic": "Briefing matin"})
    ]


def test_unknown_type_uses_output_default():
    assert RoutingPolicy(CFG).route(Event("whatever", "info")) == [
        ChannelRef("zulip", {"stream": "s", "topic": "Divers"})
    ]


def test_urgent_multi_channel_in_order():
    routes = RoutingPolicy(CFG).route(Event("email_triage", "urgent"))
    assert [c.channel for c in routes] == ["telegram", "zulip"]
    assert routes[0].target == {"bot": "gtd"}


def test_urgent_defaults_to_default_channel():
    assert [c.channel for c in RoutingPolicy({"default": "zulip"}).route(Event("x", "urgent"))] == ["zulip"]


def test_only_severity_filters_telegram_when_default():
    cfg = dict(CFG, default="telegram")
    assert RoutingPolicy(cfg).route(Event("x", "normal")) == []


def test_duplicates_removed():
    cfg = dict(CFG, urgent=["zulip", "zulip"])
    assert len(RoutingPolicy(cfg).route(Event("x", "urgent"))) == 1


def test_topic_templates():
    p = RoutingPolicy(CFG)
    today = dt.date.today().isoformat()
    assert p.route(Event("fitness", "normal"))[0].target["topic"] == f"Séance {today}"
    assert p.route(Event("calendar_action", "normal", title="RDV"))[0].target["topic"] == "RDV"


def test_dev_events():
    p = RoutingPolicy(CFG)
    assert p.route(Event("dev.completed")) == [ChannelRef("telegram", {"bot": "dev"})]
    assert p.route(Event("dev.custom")) == []
    assert RoutingPolicy.default().route(Event("dev.custom")) == [ChannelRef("telegram", {"bot": "dev"})]


def test_channels_property():
    assert RoutingPolicy(CFG).channels == {"zulip", "telegram"}
    assert RoutingPolicy.default().channels == {"telegram"}


def test_expand_env_recursive():
    assert expand_env({"a": ["${X}", "k"], "b": "p-${Y}"}, {"X": "1"}) == {"a": ["1", "k"], "b": "p-"}


def test_from_file_expands_env_from_env_file(tmp_path, monkeypatch):
    (tmp_path / "r.yaml").write_text("channels:\n  default: zulip\n  zulip:\n    site: ${ZSITE}\n")
    (tmp_path / ".env").write_text("ZSITE=http://z\n")
    monkeypatch.delenv("ZSITE", raising=False)
    p = RoutingPolicy.from_file(tmp_path / "r.yaml", env_file=tmp_path / ".env")
    assert p.channel_config("zulip")["site"] == "http://z"
    assert "ZSITE" not in os.environ


def test_from_file_missing_returns_default(tmp_path):
    assert RoutingPolicy.from_file(tmp_path / "nope.yaml").route(Event("x")) == [ChannelRef("telegram", {"bot": "gtd"})]
    assert RoutingPolicy.from_file(None).route(Event("x"))[0].channel == "telegram"


def test_from_file_without_channels_key_returns_default(tmp_path):
    (tmp_path / "r.yaml").write_text("other: 1\n")
    assert RoutingPolicy.from_file(tmp_path / "r.yaml").route(Event("x"))[0].channel == "telegram"


def test_example_file_parses():
    p = RoutingPolicy.from_file(Path(__file__).parent.parent / "routing.example.yaml")
    assert p.route(Event("briefing_morning", "normal"))[0].channel == "zulip"
    assert [c.channel for c in p.route(Event("x", "urgent"))] == ["telegram", "zulip"]
