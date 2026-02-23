"""WhatsApp bridge health check, auto-restart, and alerting."""

import asyncio
import logging
import socket
import subprocess

import httpx

from . import telegram
from .config import settings

logger = logging.getLogger(__name__)

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8082
TCP_TIMEOUT = 3  # seconds
RESTART_WAIT = 5  # seconds after systemctl restart


def _bridge_is_up() -> bool:
    """Check if the WhatsApp bridge is responding on its TCP port."""
    try:
        with socket.create_connection((BRIDGE_HOST, BRIDGE_PORT), timeout=TCP_TIMEOUT):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _restart_bridge() -> bool:
    """Attempt to restart the WhatsApp bridge via systemctl. Returns True if successful."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "whatsapp-bridge"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run systemctl restart: {e}")
        return False


async def _send_telegram_alert(chat_id: str, api_url: str) -> None:
    """Send a Telegram notification that the bridge is down."""
    msg = (
        "⚠️ <b>WhatsApp bridge down</b> — tentative de relance échouée.\n\n"
        "Vérifier :\n"
        "<code>sudo systemctl status whatsapp-bridge</code>\n"
        "<code>sudo journalctl -u whatsapp-bridge -n 50</code>\n\n"
        "Si session expirée (~20j) : re-scanner le QR code."
    )
    try:
        await telegram.send_message(msg, chat_id=chat_id, parse_mode="HTML", api_url=api_url)
    except Exception as e:
        logger.error(f"Failed to send WhatsApp bridge alert: {e}")


async def _create_trello_fix_card() -> None:
    """Create a Trello card in 'à faire' with repair steps. Skips if duplicate exists."""
    if not settings.trello_api_key or not settings.trello_token or not settings.trello_todo_list_id:
        logger.warning("Trello credentials not configured, skipping card creation")
        return

    card_name = "[Fix] WhatsApp bridge down"
    base_url = "https://api.trello.com/1"
    auth = {"key": settings.trello_api_key, "token": settings.trello_token}

    async with httpx.AsyncClient(timeout=15) as client:
        # Check for existing non-archived card with same name
        try:
            resp = await client.get(
                f"{base_url}/lists/{settings.trello_todo_list_id}/cards",
                params={**auth, "fields": "name"},
            )
            resp.raise_for_status()
            existing = [c for c in resp.json() if c["name"] == card_name]
            if existing:
                logger.info("Trello fix card already exists, skipping creation")
                return
        except Exception as e:
            logger.warning(f"Failed to check existing Trello cards: {e}")
            # Continue to create anyway

        # Create the card
        description = (
            "## Étapes de réparation\n\n"
            "1. `sudo systemctl status whatsapp-bridge` — vérifier l'état\n"
            "2. `sudo journalctl -u whatsapp-bridge -n 50` — lire les logs\n"
            "3. Si session expirée (>20 jours) : re-scanner le QR code\n"
            "   - `cd ~/projects/whatsapp-mcp/whatsapp-bridge && ./whatsapp-bridge`\n"
            "   - Scanner le QR affiché dans le terminal\n"
            "   - Ctrl+C puis `sudo systemctl start whatsapp-bridge`\n"
            "4. `sudo systemctl restart whatsapp-bridge` — relancer\n"
            "5. Vérifier : `curl -s http://localhost:8080/api/send -X POST -d '{}' 2>/dev/null ; echo $?`\n"
            "\n---\n*Carte créée automatiquement par le health check WhatsApp.*"
        )
        try:
            resp = await client.post(
                f"{base_url}/cards",
                params={
                    **auth,
                    "idList": settings.trello_todo_list_id,
                    "name": card_name,
                    "desc": description,
                    "pos": "top",
                },
            )
            resp.raise_for_status()
            logger.info(f"Created Trello fix card: {resp.json().get('shortUrl')}")
        except Exception as e:
            logger.error(f"Failed to create Trello fix card: {e}")


async def ensure_whatsapp_bridge(chat_id: str, api_url: str) -> bool:
    """Ensure the WhatsApp bridge is running. Returns True if bridge is up.

    Flow:
    1. TCP check → if up, return True
    2. If down → systemctl restart → wait → re-check
    3. If still down → Telegram alert + Trello card → return False
    """
    if _bridge_is_up():
        return True

    logger.warning("WhatsApp bridge is down, attempting restart...")

    if _restart_bridge():
        await asyncio.sleep(RESTART_WAIT)
        if _bridge_is_up():
            logger.info("WhatsApp bridge restarted successfully")
            return True

    # Restart failed
    logger.error("WhatsApp bridge restart failed")
    await _send_telegram_alert(chat_id, api_url)
    await _create_trello_fix_card()
    return False
