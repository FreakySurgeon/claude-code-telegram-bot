"""Manage pending calendar action state in data/pending-actions.json."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"confirmed", "cancelled", "expired"}
EXPIRE_DAYS = 7


def load_actions(path: Path) -> list[dict]:
    """Load actions from the JSON file. Returns empty list if file missing/invalid."""
    try:
        data = json.loads(path.read_text())
        return data.get("actions", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_actions(path: Path, actions: list[dict]) -> None:
    """Write actions list to the JSON file."""
    path.write_text(json.dumps({"actions": actions}, indent=2, ensure_ascii=False))


def add_action(path: Path, action: dict) -> None:
    """Append an action to the file."""
    actions = load_actions(path)
    actions.append(action)
    save_actions(path, actions)


def update_status(path: Path, action_id: str, new_status: str) -> None:
    """Update an action's status by ID."""
    actions = load_actions(path)
    now = datetime.now().isoformat()
    for action in actions:
        if action["id"] == action_id:
            action["status"] = new_status
            if new_status == "executed":
                action["executed_at"] = now
            if new_status in TERMINAL_STATUSES:
                action["resolved_at"] = now
            break
    save_actions(path, actions)


def is_duplicate(path: Path, event_id: str, prompt: str) -> bool:
    """Check if an action with the same event_id and prompt already exists (non-terminal)."""
    actions = load_actions(path)
    for action in actions:
        if (
            action["event_id"] == event_id
            and action["prompt"] == prompt
            and action["status"] not in TERMINAL_STATUSES
        ):
            return True
    return False


def cleanup_actions(path: Path) -> None:
    """Expire old actions and purge old terminal actions."""
    actions = load_actions(path)
    now = datetime.now()
    cleaned = []
    for action in actions:
        created = datetime.fromisoformat(action["created_at"])
        age_days = (now - created).days

        # Expire unresolved actions after EXPIRE_DAYS
        if action["status"] == "executed" and action.get("confirm", True) and age_days >= EXPIRE_DAYS:
            action["status"] = "expired"
            action["resolved_at"] = now.isoformat()
            logger.info(f"Expired calendar action: {action['id']} ({action['event_title']})")

        # Purge old terminal actions (based on resolved_at age)
        if action["status"] in TERMINAL_STATUSES and action.get("resolved_at"):
            resolved = datetime.fromisoformat(action["resolved_at"])
            resolved_age_days = (now - resolved).days
            if resolved_age_days >= EXPIRE_DAYS:
                logger.info(f"Purging old calendar action: {action['id']}")
                continue

        cleaned.append(action)
    save_actions(path, cleaned)


def get_pending_confirmations(path: Path) -> list[dict]:
    """Get actions awaiting user confirmation (executed + confirm=true)."""
    actions = load_actions(path)
    return [
        a for a in actions
        if a["status"] == "executed" and a.get("confirm", True)
    ]
