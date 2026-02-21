"""Tests for pending_actions module."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from claude_telegram.pending_actions import (
    load_actions,
    save_actions,
    add_action,
    update_status,
    cleanup_actions,
    is_duplicate,
    get_pending_confirmations,
)


@pytest.fixture
def tmp_actions_file(tmp_path):
    """Create a temporary pending-actions.json."""
    path = tmp_path / "pending-actions.json"
    path.write_text('{"actions": []}')
    return path


@pytest.fixture
def sample_action():
    return {
        "id": "evt_abc123_0",
        "event_id": "abc123",
        "event_title": "Test Event",
        "event_date": "2026-02-22",
        "prompt": "Do something",
        "confirm": True,
        "status": "pending",
        "thread_id": 456,
        "created_at": "2026-02-21T05:30:00",
        "executed_at": None,
        "resolved_at": None,
    }


def test_load_empty(tmp_actions_file):
    actions = load_actions(tmp_actions_file)
    assert actions == []


def test_load_missing_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    actions = load_actions(path)
    assert actions == []


def test_add_and_load(tmp_actions_file, sample_action):
    add_action(tmp_actions_file, sample_action)
    actions = load_actions(tmp_actions_file)
    assert len(actions) == 1
    assert actions[0]["id"] == "evt_abc123_0"


def test_update_status(tmp_actions_file, sample_action):
    add_action(tmp_actions_file, sample_action)
    update_status(tmp_actions_file, "evt_abc123_0", "executed")
    actions = load_actions(tmp_actions_file)
    assert actions[0]["status"] == "executed"
    assert actions[0]["executed_at"] is not None


def test_update_status_confirmed(tmp_actions_file, sample_action):
    sample_action["status"] = "executed"
    add_action(tmp_actions_file, sample_action)
    update_status(tmp_actions_file, "evt_abc123_0", "confirmed")
    actions = load_actions(tmp_actions_file)
    assert actions[0]["status"] == "confirmed"
    assert actions[0]["resolved_at"] is not None


def test_is_duplicate(tmp_actions_file, sample_action):
    add_action(tmp_actions_file, sample_action)
    assert is_duplicate(tmp_actions_file, "abc123", "Do something") is True
    assert is_duplicate(tmp_actions_file, "abc123", "Different prompt") is False
    assert is_duplicate(tmp_actions_file, "xyz789", "Do something") is False


def test_is_duplicate_ignores_terminal(tmp_actions_file, sample_action):
    sample_action["status"] = "confirmed"
    add_action(tmp_actions_file, sample_action)
    assert is_duplicate(tmp_actions_file, "abc123", "Do something") is False


def test_cleanup_expires_old(tmp_actions_file, sample_action):
    sample_action["status"] = "executed"
    sample_action["created_at"] = (datetime.now() - timedelta(days=8)).isoformat()
    add_action(tmp_actions_file, sample_action)
    cleanup_actions(tmp_actions_file)
    actions = load_actions(tmp_actions_file)
    assert actions[0]["status"] == "expired"


def test_cleanup_purges_old_terminal(tmp_actions_file, sample_action):
    sample_action["status"] = "confirmed"
    sample_action["created_at"] = (datetime.now() - timedelta(days=8)).isoformat()
    sample_action["resolved_at"] = (datetime.now() - timedelta(days=8)).isoformat()
    add_action(tmp_actions_file, sample_action)
    cleanup_actions(tmp_actions_file)
    actions = load_actions(tmp_actions_file)
    assert len(actions) == 0


def test_get_pending_confirmations(tmp_actions_file, sample_action):
    sample_action["status"] = "executed"
    add_action(tmp_actions_file, sample_action)
    pending = get_pending_confirmations(tmp_actions_file)
    assert len(pending) == 1
    assert pending[0]["id"] == "evt_abc123_0"


def test_get_pending_confirmations_excludes_no_confirm(tmp_actions_file, sample_action):
    sample_action["status"] = "executed"
    sample_action["confirm"] = False
    add_action(tmp_actions_file, sample_action)
    pending = get_pending_confirmations(tmp_actions_file)
    assert len(pending) == 0
