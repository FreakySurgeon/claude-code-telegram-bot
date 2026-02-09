"""FastAPI application - Telegram webhook handler."""

import asyncio
import logging
import random
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request

from .bots import BotConfig, create_bots
from .transcribe import transcribe_audio

# Claude Code spinner words (from the CLI)
# Source: https://github.com/levindixon/tengu_spinner_words
SPINNER_VERBS = [
    "Accomplishing", "Actioning", "Actualizing", "Baking", "Booping", "Brewing",
    "Calculating", "Cerebrating", "Channelling", "Churning", "Clauding", "Coalescing",
    "Cogitating", "Combobulating", "Computing", "Concocting", "Conjuring", "Considering",
    "Contemplating", "Cooking", "Crafting", "Creating", "Crunching", "Deciphering",
    "Deliberating", "Determining", "Discombobulating", "Divining", "Doing", "Effecting",
    "Elucidating", "Enchanting", "Envisioning", "Finagling", "Flibbertigibbeting",
    "Forging", "Forming", "Frolicking", "Generating", "Germinating", "Hatching",
    "Herding", "Honking", "Hustling", "Ideating", "Imagining", "Incubating", "Inferring",
    "Jiving", "Manifesting", "Marinating", "Meandering", "Moseying", "Mulling",
    "Mustering", "Musing", "Noodling", "Percolating", "Perusing", "Philosophising",
    "Pondering", "Pontificating", "Processing", "Puttering", "Puzzling", "Reticulating",
    "Ruminating", "Scheming", "Schlepping", "Shimmying", "Shucking", "Simmering",
    "Smooshing", "Spelunking", "Spinning", "Stewing", "Sussing", "Synthesizing",
    "Thinking", "Tinkering", "Transmuting", "Unfurling", "Unravelling", "Vibing",
    "Wandering", "Whirring", "Wibbling", "Wizarding", "Working", "Wrangling",
]

CRON_PROMPTS = {
    "morning": (
        "C'est le rappel automatique du matin (6h).\n\n"
        "Effectue ces actions :\n"
        "1. Récupère l'heure actuelle avec get-current-time\n"
        "2. Lis l'agenda Google Calendar pour aujourd'hui (calendriers: chauvet.t@gmail.com, Planning, Gardes HE)\n"
        "3. Lis les tâches Trello dans 'En cours' et 'à faire'\n"
        "4. Vérifie s'il y a des tâches en attente depuis plus de 7 jours\n\n"
        "Génère UNIQUEMENT un message de mise au point (pas d'autres actions) avec ce format :\n\n"
        "☀️ **Mise au point - [Jour] [Date]**\n\n"
        "📅 **Programme du jour :**\n• [Liste des RDV avec heures]\n\n"
        "✅ **Focus du jour :**\n• [Tâches En cours]\n• [2-3 priorités de 'à faire']\n\n"
        "⚠️ **Rappels :**\n• [Tâches en attente à relancer si > 7j]\n\n"
        "💪 [Message motivant adapté au jour - mardi=bloc, vendredi=admin, etc.]\n\n"
        "Réponds ici si tu veux ajuster quelque chose !"
    ),
    "evening": (
        "C'est le rappel automatique du soir (18h).\n\n"
        "Effectue ces actions :\n"
        "1. Récupère l'heure actuelle avec get-current-time\n"
        "2. Lis l'agenda Google Calendar pour demain\n"
        "3. Lis les tâches Trello : 'En cours', 'Inbox' (non traitées)\n"
        "4. Vérifie le jour de la semaine (si vendredi/samedi → rappel weekly review)\n\n"
        "Génère UNIQUEMENT un message de bilan (pas d'autres actions) avec ce format :\n\n"
        "🌙 **Bilan de fin de journée**\n\n"
        "📋 **État des tâches :**\n• En cours : [X] tâches\n• Inbox à trier : [Y] éléments\n\n"
        "📅 **Demain :**\n• [Premier RDV ou événement important]\n• [Résumé du programme]\n\n"
        "🎯 **Actions suggérées :**\n• [1-2 suggestions : trier inbox, préparer demain, etc.]\n\n"
        "[Si vendredi/samedi: 📊 N'oublie pas la weekly review ce week-end !]\n\n"
        "Comment s'est passée ta journée ?"
    ),
    "weekly": (
        "C'est le rappel automatique de weekly review (dimanche 10h).\n\n"
        "Effectue ces actions :\n"
        "1. Récupère l'heure actuelle\n"
        "2. Lis toutes les listes Trello : Inbox, En attente, à faire, En cours, Si du temps...\n"
        "3. Compte les tâches par liste\n"
        "4. Identifie les tâches 'En attente' les plus anciennes\n\n"
        "Génère UNIQUEMENT un message de weekly review avec ce format :\n\n"
        "📋 **Weekly Review GTD - Dimanche**\n\n"
        "📊 **État du board Trello :**\n"
        "• Inbox : [X] éléments à trier\n"
        "• En cours : [X] tâches (max 3 recommandé)\n"
        "• À faire : [X] prochaines actions\n"
        "• En attente : [X] tâches (⚠️ si > 5)\n"
        "• Someday : [X] idées en réserve\n\n"
        "⚠️ **Points d'attention :**\n"
        "• [Tâches en attente > 7 jours]\n"
        "• [Inbox non vidée]\n\n"
        "✅ **Checklist GTD :**\n"
        "☐ Vider l'Inbox (traiter chaque élément)\n"
        "☐ Revoir 'En attente' - relancer si nécessaire\n"
        "☐ Parcourir 'Si du temps...' - réactiver ?\n"
        "☐ Vérifier l'agenda de la semaine à venir\n"
        "☐ Identifier les 3 priorités de la semaine\n\n"
        "Veux-tu qu'on fasse la review ensemble maintenant ?"
    ),
}

def get_thinking_message() -> str:
    """Get a random thinking message with emoji."""
    verb = random.choice(SPINNER_VERBS)
    return f"✨ <i>{verb}...</i>"

def get_continue_message() -> str:
    """Get a random continue message with emoji."""
    verb = random.choice(SPINNER_VERBS)
    return f"🔄 <i>{verb}...</i>"

from . import telegram
from .claude import sessions, ClaudeResult, PermissionDenial, get_session_permission_mode
from .config import settings
from .markdown import markdown_to_telegram_html
from .tunnel import tunnel, CloudflareTunnel
from .queue import QueueItem, RequestQueue, process_queue_item

# Store pending permission requests for retry
pending_permissions: dict[str, dict] = {}  # chat_id -> {message, denials, session_key, bot_name}

# Bot configurations (initialized at startup)
bots: dict[str, BotConfig] = {}

# Map chat_id -> bot_name for routing notifications
chat_to_bot: dict[str, str] = {}

# Polling tasks
polling_tasks: list[asyncio.Task] = []

# Queue for GTD bot (initialized in lifespan)
gtd_queue: RequestQueue | None = None
queue_worker_task: asyncio.Task | None = None

def get_runner_for_bot(bot: BotConfig):
    """Get the appropriate runner for a bot."""
    if bot.fixed_working_dir:
        return sessions.get_session(bot.fixed_working_dir)
    return sessions.get_current_session()


def build_session_buttons(session_list: list, current) -> dict:
    """Build inline keyboard buttons for session selection."""
    buttons = []
    row = []
    for i, (dir_key, session) in enumerate(session_list, 1):
        # Mark current session with checkmark
        label = f"{'✓ ' if session == current else ''}{i}. {session.short_name}"
        row.append({"text": label, "callback_data": f"dir:{dir_key}"})
        # Max 2 buttons per row
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return {"inline_keyboard": buttons}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Current tunnel URL (if using tunnel mode)
tunnel_url: str | None = None


async def poll_updates(bot: BotConfig):
    """Poll Telegram for updates for a specific bot."""
    offset = 0
    logger.info(f"Starting polling for bot '{bot.name}'...")

    while True:
        try:
            updates = await telegram.get_updates(offset=offset, timeout=30, api_url=bot.api_url)

            for update in updates:
                offset = update["update_id"] + 1

                if "message" in update:
                    await handle_message(update["message"], bot)
                elif "callback_query" in update:
                    await handle_callback(update["callback_query"], bot)

        except asyncio.CancelledError:
            logger.info(f"Polling stopped for bot '{bot.name}'")
            break
        except Exception as e:
            logger.error(f"Polling error ({bot.name}): {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and teardown."""
    global bots, chat_to_bot, polling_tasks, tunnel_url, gtd_queue, queue_worker_task

    # Initialize bots
    bots = create_bots()
    for bot_name, bot in bots.items():
        chat_to_bot[str(bot.chat_id)] = bot_name
    logger.info(f"Initialized bots: {list(bots.keys())}")

    mode = settings.mode

    # Tunnel mode — only for dev bot
    if mode == "tunnel":
        if not CloudflareTunnel.is_available():
            logger.warning("cloudflared not found, falling back to polling mode")
            mode = "polling"
        else:
            logger.info("Starting Cloudflare tunnel...")
            tunnel.port = settings.port
            tunnel_url = await tunnel.start()

            if tunnel_url:
                webhook_url = f"{tunnel_url}{settings.webhook_path}"
                logger.info(f"Tunnel URL: {tunnel_url}")
                logger.info(f"Setting webhook: {webhook_url}")
                try:
                    await telegram.set_webhook_with_retry(webhook_url, api_url=bots["dev"].api_url)
                    logger.info("Webhook set successfully")
                except Exception as e:
                    logger.error(f"Webhook setup failed after retries: {e}, falling back to polling")
                    mode = "polling"
            else:
                logger.warning("Tunnel failed to start, falling back to polling mode")
                mode = "polling"

    # Manual webhook mode
    if mode == "webhook" and settings.webhook_url:
        webhook_url = f"{settings.webhook_url}{settings.webhook_path}"
        logger.info(f"Setting webhook: {webhook_url}")
        await telegram.set_webhook(webhook_url, api_url=bots["dev"].api_url)

    # Polling mode (fallback or default)
    if mode == "polling":
        logger.info("Starting polling mode...")
        for bot_name, bot in bots.items():
            await telegram.delete_webhook(api_url=bot.api_url)
            task = asyncio.create_task(poll_updates(bot))
            polling_tasks.append(task)

    # Start GTD queue worker
    gtd_bot_instance = bots.get("gtd")
    if gtd_bot_instance:
        gtd_queue = RequestQueue(maxsize=10)
        queue_worker_task = asyncio.create_task(queue_worker(gtd_queue, gtd_bot_instance))

    yield

    # Cleanup
    if queue_worker_task:
        queue_worker_task.cancel()
        try:
            await queue_worker_task
        except asyncio.CancelledError:
            pass

    for task in polling_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if tunnel.is_running:
        await telegram.delete_webhook(api_url=bots["dev"].api_url)
        await tunnel.stop()

    if mode == "webhook" and settings.webhook_url:
        await telegram.delete_webhook(api_url=bots["dev"].api_url)



async def queue_worker(queue: RequestQueue, bot: BotConfig):
    """Worker loop: dequeue and process items one at a time."""
    logger.info("Queue worker started")
    while True:
        try:
            item = await queue.dequeue()
            logger.info(f"Processing queued {item.source} request (retry={item.retry_count})")
            runner = get_runner_for_bot(bot)
            await process_queue_item(item, runner, bot, queue=queue)
        except asyncio.CancelledError:
            logger.info("Queue worker stopped")
            break
        except Exception:
            logger.exception("Queue worker error")
            await asyncio.sleep(1)


app = FastAPI(title="Claude Telegram", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    current = sessions.get_current_session()
    return {
        "status": "ok",
        "claude_running": sessions.any_running(),
        "current_session": current.short_name,
        "in_conversation": current.is_in_conversation(),
        "active_sessions": len(sessions.sessions),
        "queue_size": gtd_queue.size if gtd_queue else 0,
    }


@app.post(settings.webhook_path)
async def webhook(request: Request):
    """Handle Telegram webhook updates (dev bot only in tunnel/webhook mode)."""
    data = await request.json()
    logger.info(f"Received update: {data}")

    dev_bot = bots.get("dev")
    if not dev_bot:
        return {"ok": False}

    if "message" in data:
        await handle_message(data["message"], dev_bot)
    elif "callback_query" in data:
        await handle_callback(data["callback_query"], dev_bot)

    return {"ok": True}


async def handle_message(message: dict, bot: BotConfig):
    """Process incoming Telegram message."""
    chat_id = str(message["chat"]["id"])

    if not bot.is_authorized(chat_id):
        logger.warning(f"Unauthorized access from chat_id: {chat_id} on bot {bot.name}")
        return

    # Handle voice messages
    voice = message.get("voice") or message.get("audio")
    if voice:
        await handle_voice(message, bot)
        return

    text = message.get("text", "")
    if not text:
        return

    # Handle commands
    if text.startswith("/"):
        await handle_command(text, chat_id, bot)
        return

    # Check if it's a quick reply
    quick_reply = is_quick_reply(text)

    # Auto-continue if current session is in conversation
    runner = get_runner_for_bot(bot)
    continue_session = runner.is_in_conversation() or quick_reply

    # GTD bot: enqueue for sequential processing
    if not bot.multi_session and gtd_queue is not None:
        item = QueueItem(
            prompt=text,
            source="telegram",
            chat_id=chat_id,
            continue_session=continue_session,
        )
        added = await gtd_queue.enqueue(item)
        if not added:
            await telegram.send_message(
                "⚠️ Queue pleine (10 max), réessaie plus tard",
                chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            )
        elif gtd_queue.size > 1:
            await telegram.send_message(
                f"📥 Message reçu (position {gtd_queue.size} dans la file)",
                chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            )
        return

    # Dev bot: direct execution (existing behavior)
    await run_claude(text, chat_id, bot, continue_session=continue_session)


async def handle_voice(message: dict, bot: BotConfig):
    """Handle voice/audio messages — transcribe and offer to process."""
    chat_id = str(message["chat"]["id"])
    voice = message.get("voice") or message.get("audio")
    file_id = voice["file_id"]

    await telegram.send_message(
        "🎤 <i>Transcription en cours...</i>",
        chat_id=chat_id,
        parse_mode="HTML",
        api_url=bot.api_url,
    )

    try:
        # Download file from Telegram
        file_info = await telegram.get_file(file_id, api_url=bot.api_url)
        file_path = file_info["result"]["file_path"]
        audio_data = await telegram.download_file(file_path, api_url=bot.api_url)

        # Save to temp file
        import tempfile
        suffix = Path(file_path).suffix or ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # Transcribe
        result = await transcribe_audio(tmp_path)

        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

        # For GTD bot: process directly via Claude
        if not bot.multi_session:
            transcription_prompt = f"[Transcription vocale ({result.duration_formatted}, {result.engine})]\n\n{result.text}"
            if gtd_queue is not None:
                item = QueueItem(
                    prompt=transcription_prompt,
                    source="telegram",
                    chat_id=chat_id,
                    continue_session=False,
                )
                added = await gtd_queue.enqueue(item)
                if not added:
                    await telegram.send_message(
                        "⚠️ Queue pleine (10 max), réessaie plus tard",
                        chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
                    )
                elif gtd_queue.size > 1:
                    await telegram.send_message(
                        f"📥 Message vocal reçu (position {gtd_queue.size} dans la file)",
                        chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
                    )
            else:
                await run_claude(transcription_prompt, chat_id, bot, continue_session=False)
        else:
            # For Dev bot: show transcription with button to process
            # Truncate text for callback_data (max 64 bytes)
            truncated = result.text[:180] if len(result.text) > 180 else result.text
            buttons = {"inline_keyboard": [[
                {"text": "✅ Send to Claude", "callback_data": f"voice:{truncated}"},
            ]]}
            await telegram.send_message(
                f"🎤 <b>Transcription</b> ({result.duration_formatted})\n\n{result.text}",
                chat_id=chat_id,
                parse_mode="HTML",
                reply_markup=buttons,
                api_url=bot.api_url,
            )

    except Exception as e:
        logger.exception("Transcription error")
        await telegram.send_message(
            f"❌ Transcription failed: <code>{e}</code>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )


def is_quick_reply(text: str) -> bool:
    """Check if the message is a quick reply (number, yes/no, etc.)."""
    text = text.strip().lower()
    # Single number
    if re.match(r"^\d+$", text):
        return True
    # Common quick replies
    if text in ("yes", "no", "y", "n", "ok", "cancel", "skip", "done", "next"):
        return True
    return False


async def handle_command(text: str, chat_id: str, bot: BotConfig):
    """Handle bot commands."""
    cmd = text.split()[0].lower()
    args = text[len(cmd):].strip()

    # Check command whitelist for this bot
    if cmd not in bot.commands_whitelist:
        await telegram.send_message(
            f"Commande inconnue — tape <code>/help</code> pour voir les commandes",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )
        return

    if cmd == "/start" or cmd == "/help":
        if bot.multi_session:
            await telegram.send_message(
                "<b>Claude Code</b> via Telegram\n\n"
                "<b>Commands</b>\n"
                "<code>/c &lt;msg&gt;</code> — Continue conversation\n"
                "<code>/new &lt;msg&gt;</code> — Fresh session\n"
                "<code>/dir path</code> — Switch directory (relative to ~)\n"
                "<code>/dirs</code> — List sessions + buttons\n"
                "<code>/repos</code> — Favorite repos\n"
                "<code>/rmdir path</code> — Remove a session\n"
                "<code>/compact</code> — Compact context\n"
                "<code>/cancel</code> — Stop current task\n"
                "<code>/status</code> — Check status\n\n"
                "<b>Tips</b>\n"
                "• Just type to chat — auto-continues for 10 min\n"
                "• <code>/dir projects/foo</code> = ~/projects/foo\n"
                "• Tap buttons in /repos to start in a repo",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
        else:
            await telegram.send_message(
                "<b>Assistant GTD</b> via Telegram\n\n"
                "<b>Commands</b>\n"
                "<code>/new &lt;msg&gt;</code> — Nouveau sujet\n"
                "<code>/compact</code> — Compacter le contexte\n"
                "<code>/cancel</code> — Arrêter la tâche\n"
                "<code>/status</code> — Vérifier le statut\n\n"
                "<b>Tips</b>\n"
                "• Écris ou envoie un vocal — je comprends tout\n"
                "• La conversation continue automatiquement",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )

    elif cmd == "/c" or cmd == "/continue":
        if args:
            await run_claude(args, chat_id, bot, continue_session=True)
        else:
            await telegram.send_message(
                "Usage: <code>/c &lt;message&gt;</code>",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )

    elif cmd == "/new":
        if args:
            # Reset current session's conversation state
            runner = get_runner_for_bot(bot)
            runner.last_interaction = None
            await run_claude(args, chat_id, bot, continue_session=False)
        else:
            await telegram.send_message(
                "Usage: <code>/new &lt;message&gt;</code>",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )

    elif cmd == "/dir":
        if args:
            session = sessions.switch_session(args)
            status = "🔄 running" if session.is_running else "💤 idle"
            conv = "in conversation" if session.is_in_conversation() else "fresh"

            # Check for stored session context
            context = None
            if not session.context_shown and not session.is_in_conversation():
                context = session.get_session_context()

            msg = f"📂 Switched to <code>{session.short_name}</code>\nStatus: {status} • {conv}"
            if context:
                msg += f"\n\n📜 <b>Previous context:</b>\n<i>{context}</i>"

            await telegram.send_message(
                msg,
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
        else:
            # Show session picker if sessions exist
            session_list = sessions.list_sessions()
            current = get_runner_for_bot(bot)
            if len(session_list) > 1:
                buttons = build_session_buttons(session_list, current)
                await telegram.send_message(
                    f"📂 Current: <code>{current.short_name}</code>\n\n"
                    "Select or add new: <code>/dir projects/foo</code>\n"
                    "<i>(paths are relative to home)</i>",
                    chat_id=chat_id,
                    parse_mode="HTML",
                    reply_markup=buttons,
                    api_url=bot.api_url,
                )
            else:
                await telegram.send_message(
                    f"📂 Current: <code>{current.short_name}</code>\n\n"
                    "Usage: <code>/dir projects/foo</code>\n"
                    "<i>(paths are relative to home)</i>",
                    chat_id=chat_id,
                    parse_mode="HTML",
                    api_url=bot.api_url,
                )

    elif cmd == "/dirs":
        session_list = sessions.list_sessions()
        if not session_list:
            await telegram.send_message(
                "No active sessions",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
        else:
            current = get_runner_for_bot(bot)
            lines = ["<b>Active Sessions</b>\n"]
            for i, (dir_key, session) in enumerate(session_list, 1):
                is_current = session == current
                if session.is_running:
                    status = "🔄"  # Running
                elif is_current:
                    status = "📍"  # Current/selected
                else:
                    status = "💤"  # Idle
                lines.append(f"{i}. {status} <code>{session.short_name}</code>")
            buttons = build_session_buttons(session_list, current)
            await telegram.send_message(
                "\n".join(lines),
                chat_id=chat_id,
                parse_mode="HTML",
                reply_markup=buttons,
                api_url=bot.api_url,
            )

    elif cmd == "/compact":
        runner = get_runner_for_bot(bot)
        if runner.is_running:
            await telegram.send_message(
                "⏳ Claude is busy — use <code>/cancel</code> first",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
            return
        await telegram.send_message(
            f"🗜 <i>Compacting context for {runner.short_name}...</i>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )
        result = await runner.compact()
        await send_response(result.text, chat_id, api_url=bot.api_url)

    elif cmd == "/cancel":
        runner = get_runner_for_bot(bot)
        cancelled = await runner.cancel()
        drained = 0
        if gtd_queue and not bot.multi_session:
            drained = gtd_queue.drain()
        if cancelled or drained:
            msg = f"🛑 Cancelled <code>{runner.short_name}</code>"
            if drained:
                msg += f" + {drained} en file supprimé(s)"
            await telegram.send_message(msg, chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url)
        else:
            await telegram.send_message("Nothing to cancel", chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url)

    elif cmd == "/status":
        runner = get_runner_for_bot(bot)
        if runner.is_running:
            status = "🔄 <b>Running</b>"
        else:
            status = "💤 <b>Idle</b>"
        conv = "in conversation" if runner.is_in_conversation() else "new session"
        msg = f"📂 <code>{runner.short_name}</code>\n{status} • {conv}"
        if gtd_queue and not bot.multi_session:
            msg += f"\n📥 Queue: {gtd_queue.size} en attente"
        await telegram.send_message(msg, chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url)

    elif cmd == "/rmdir":
        if args:
            if sessions.remove_session(args):
                current = get_runner_for_bot(bot)
                await telegram.send_message(
                    f"🗑 Removed session <code>{args}</code>\n"
                    f"📍 Current: <code>{current.short_name}</code>",
                    chat_id=chat_id,
                    parse_mode="HTML",
                    api_url=bot.api_url,
                )
            else:
                await telegram.send_message(
                    f"❌ Could not remove <code>{args}</code>\n"
                    "<i>(Session not found or currently running)</i>",
                    chat_id=chat_id,
                    parse_mode="HTML",
                    api_url=bot.api_url,
                )
        else:
            await telegram.send_message(
                "Usage: <code>/rmdir path</code>",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )

    elif cmd == "/repos":
        favorites = settings.get_favorite_repos()
        if not favorites:
            await telegram.send_message(
                "No favorite repos configured.\n\n"
                "Add <code>FAVORITE_REPOS</code> to your .env:\n"
                "<code>FAVORITE_REPOS=projects/foo,projects/bar</code>",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
        else:
            # Build buttons for favorite repos
            current = get_runner_for_bot(bot)
            buttons = []
            row = []
            for repo in favorites:
                # Use last part of path as label
                label = repo.split("/")[-1]
                row.append({"text": f"📁 {label}", "callback_data": f"repo:{repo}"})
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)

            await telegram.send_message(
                f"<b>Favorite Repos</b>\n"
                f"📍 Current: <code>{current.short_name}</code>\n\n"
                "Select a repo to switch:",
                chat_id=chat_id,
                parse_mode="HTML",
                reply_markup={"inline_keyboard": buttons},
                api_url=bot.api_url,
            )

    else:
        # Unknown command - maybe they meant to chat?
        await telegram.send_message(
            f"Unknown command — try <code>/c {text}</code> to continue",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )


async def handle_callback(callback: dict, bot: BotConfig):
    """Handle callback query from inline buttons."""
    query_id = callback["id"]
    data = callback.get("data", "")
    chat_id = callback["message"]["chat"]["id"]

    logger.info(f"handle_callback: data={data}, chat_id={chat_id}")

    if not bot.is_authorized(chat_id):
        logger.warning(f"Unauthorized callback from {chat_id}")
        return

    # Answer the callback to remove loading state
    await telegram.answer_callback(query_id, api_url=bot.api_url)

    if data.startswith("reply:"):
        reply = data[6:]  # Remove "reply:" prefix
        await run_claude(reply, str(chat_id), bot, continue_session=True)

    elif data.startswith("voice:"):
        voice_text = data[6:]
        await run_claude(voice_text, str(chat_id), bot, continue_session=False)

    elif data.startswith("dir:") or data.startswith("repo:"):
        # Handle both dir: and repo: callbacks the same way
        dir_path = data.split(":", 1)[1]  # Remove prefix
        session = sessions.switch_session(dir_path)
        status = "🔄 running" if session.is_running else "💤 idle"
        conv = "in conversation" if session.is_in_conversation() else "fresh"

        # Check for stored session context
        context = None
        if not session.context_shown and not session.is_in_conversation():
            context = session.get_session_context()

        msg = f"📂 Switched to <code>{session.short_name}</code>\nStatus: {status} • {conv}"
        if context:
            msg += f"\n\n📜 <b>Previous context:</b>\n<i>{context}</i>"

        await telegram.send_message(
            msg,
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )

    elif data == "perm:allow":
        # User approved the permission request - retry with allowed tools
        logger.info(f"perm:allow clicked, pending_permissions: {pending_permissions}")
        pending = pending_permissions.get(str(chat_id))
        if not pending:
            await telegram.send_message(
                "No pending permission request.",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
            return

        # Build allowed tools list from denials
        # Format: "Tool" or "Bash(pattern:*)" for command matching
        allowed_tools = []
        for denial in pending["denials"]:
            tool = denial.tool_name
            tool_input = denial.tool_input
            if tool in ("Write", "Edit", "Read"):
                # For file tools, just allow the tool (can't filter by path)
                allowed_tools.append(tool)
            elif tool == "Bash":
                # For Bash, try to match the specific command
                cmd = tool_input.get("command", "")
                # Extract first word of command for pattern matching
                first_word = cmd.split()[0] if cmd.split() else ""
                if first_word:
                    allowed_tools.append(f"Bash({first_word}:*)")
                else:
                    allowed_tools.append("Bash")
            else:
                allowed_tools.append(tool)

        # Clear pending and retry
        original_message = pending["message"]
        del pending_permissions[str(chat_id)]

        await telegram.send_message(
            f"✅ <i>Retrying with permissions...</i>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )

        await run_claude(
            original_message,
            str(chat_id),
            bot,
            continue_session=True,
            allowed_tools=allowed_tools,
        )

    elif data == "perm:deny":
        # User denied - just clear the pending request
        if str(chat_id) in pending_permissions:
            del pending_permissions[str(chat_id)]
        await telegram.send_message(
            "❌ Permission denied. Request cancelled.",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )

    elif data == "perm:bypass":
        # User wants to continue with bypass permissions
        logger.info(f"perm:bypass clicked, pending_permissions: {pending_permissions}")
        pending = pending_permissions.get(str(chat_id))
        if not pending:
            await telegram.send_message(
                "No pending permission request.",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )
            return

        # Clear pending and retry with bypass
        original_message = pending["message"]
        del pending_permissions[str(chat_id)]

        await telegram.send_message(
            f"🔓 <i>Retrying with bypass permissions...</i>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )

        # Retry with bypass permissions
        await run_claude(original_message, str(chat_id), bot, continue_session=True, bypass_permissions=True)


async def animate_status(chat_id: str, message_id: int, continue_session: bool, session_name: str, api_url: str | None = None):
    """Animate the status message with rotating messages."""
    prefix = f"[<code>{session_name}</code>] " if session_name != "default" else ""
    try:
        while True:
            await asyncio.sleep(2.5)  # Update every 2.5 seconds
            status = get_continue_message() if continue_session else get_thinking_message()
            new_status = f"{prefix}{status}"
            try:
                await telegram.edit_message(message_id, new_status, chat_id, parse_mode="HTML", api_url=api_url)
            except Exception:
                pass  # Ignore edit errors (message may be deleted)
    except asyncio.CancelledError:
        pass


async def run_claude(
    message: str,
    chat_id: str,
    bot: BotConfig,
    continue_session: bool = False,
    allowed_tools: list[str] | None = None,
    bypass_permissions: bool = False,
):
    """Run Claude and send response to Telegram."""
    # GTD bot always bypasses permissions
    if not bot.multi_session:
        bypass_permissions = True

    runner = get_runner_for_bot(bot)
    session_name = runner.short_name
    prefix = f"[<code>{session_name}</code>] " if session_name != "default" else ""

    if runner.is_running:
        await telegram.send_message(
            f"{prefix}⏳ Claude is busy — use <code>/cancel</code> to stop",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )
        return

    # Check for stored session context on first interaction
    if not runner.context_shown and not runner.is_in_conversation():
        context = runner.get_session_context()
        if context:
            await telegram.send_message(
                f"{prefix}📜 <b>Resuming previous session:</b>\n<i>{context}</i>",
                chat_id=chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )

    # Send animated status message
    initial_status = get_continue_message() if continue_session else get_thinking_message()
    status_msg = await telegram.send_message(
        f"{prefix}{initial_status}",
        chat_id=chat_id,
        parse_mode="HTML",
        api_url=bot.api_url,
    )
    message_id = status_msg.get("result", {}).get("message_id")

    # Start animation task
    animation_task = None
    if message_id:
        animation_task = asyncio.create_task(
            animate_status(chat_id, message_id, continue_session, session_name, api_url=bot.api_url)
        )

    try:
        result = await runner.run(
            message,
            continue_session=continue_session,
            allowed_tools=allowed_tools,
            bypass_permissions=bypass_permissions,
            system_prompt=bot.system_prompt,
            mcp_config=bot.mcp_config_path,
        )

        # Stop animation
        if animation_task:
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass

        # Delete status message
        await telegram.delete_message(chat_id, message_id, api_url=bot.api_url)

        # Check for permission denials
        logger.info(f"Result: text={result.text[:100] if result.text else 'None'}, denials={result.permission_denials}")
        if result.permission_denials:
            await send_permission_request(
                result, message, chat_id, session_name, sessions.current_dir, bot
            )
        else:
            await send_response(result.text, chat_id, session_name=session_name, api_url=bot.api_url)

    except Exception as e:
        # Stop animation on error
        if animation_task:
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass
        if message_id:
            await telegram.delete_message(chat_id, message_id, api_url=bot.api_url)

        logger.exception("Claude error")
        await telegram.send_message(
            f"{prefix}❌ <b>Error:</b> <code>{e}</code>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=bot.api_url,
        )


async def send_permission_request(
    result: ClaudeResult,
    original_message: str,
    chat_id: str,
    session_name: str,
    session_dir: str,
    bot: BotConfig,
):
    """Send permission denial info to user with Allow/Deny buttons."""
    prefix = f"[<code>{session_name}</code>] " if session_name != "default" else ""

    # Format the denied permissions
    denial_lines = []
    for d in result.permission_denials:
        tool = d.tool_name
        if tool == "Write":
            path = d.tool_input.get("file_path", "unknown")
            denial_lines.append(f"• <b>Write</b> to <code>{path}</code>")
        elif tool == "Bash":
            cmd = d.tool_input.get("command", "unknown")[:60]
            denial_lines.append(f"• <b>Bash</b>: <code>{cmd}</code>")
        elif tool == "Edit":
            path = d.tool_input.get("file_path", "unknown")
            denial_lines.append(f"• <b>Edit</b> <code>{path}</code>")
        elif tool == "Read":
            path = d.tool_input.get("file_path", "unknown")
            denial_lines.append(f"• <b>Read</b> <code>{path}</code>")
        else:
            denial_lines.append(f"• <b>{tool}</b>: {str(d.tool_input)[:50]}")

    # Store pending request for retry
    pending_permissions[str(chat_id)] = {
        "message": original_message,
        "denials": result.permission_denials,
        "session_dir": session_dir,
        "bot_name": bot.name,
    }

    # Build message with buttons
    msg = (
        f"{prefix}⚠️ <b>Permission denied:</b>\n"
        + "\n".join(denial_lines)
    )

    # Also show partial result if any
    if result.text.strip():
        msg += f"\n\n<i>{result.text[:500]}</i>"

    # Check if original session was in bypass mode
    permission_mode = get_session_permission_mode(session_dir)
    was_bypass = permission_mode == "bypassPermissions"

    # Build buttons - add bypass option if session was originally in bypass mode
    button_row = [
        {"text": "✅ Allow & Retry", "callback_data": "perm:allow"},
        {"text": "❌ Deny", "callback_data": "perm:deny"},
    ]

    buttons = {"inline_keyboard": [button_row]}

    # Add bypass button on second row if session was in bypass mode
    if was_bypass:
        buttons["inline_keyboard"].append([
            {"text": "🔓 Continue with bypass", "callback_data": "perm:bypass"}
        ])
        msg += "\n\n<i>Original session was in bypass mode.</i>"

    await telegram.send_message(
        msg,
        chat_id=chat_id,
        parse_mode="HTML",
        reply_markup=buttons,
        api_url=bot.api_url,
    )


async def send_response(text: str, chat_id: str, chunk_size: int = 4000, session_name: str = "default", api_url: str | None = None):
    """Send Claude's response, with quick-reply buttons if numbered options detected."""
    if not text.strip():
        await telegram.send_message(
            "<i>(no output)</i>",
            chat_id=chat_id,
            parse_mode="HTML",
            api_url=api_url,
        )
        return

    # Detect numbered options before converting to HTML
    buttons = detect_options(text)

    # Convert markdown to Telegram HTML
    html_text = markdown_to_telegram_html(text)

    # Split into chunks if needed
    chunks = split_text(html_text, chunk_size)

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        reply_markup = buttons if (is_last and buttons) else None
        try:
            await telegram.send_message(
                chunk,
                chat_id=chat_id,
                parse_mode="HTML",
                reply_markup=reply_markup,
                api_url=api_url,
            )
        except Exception as e:
            # Fallback to plain text if HTML fails
            logger.warning(f"HTML parse failed, falling back to plain text: {e}")
            await telegram.send_message(
                text if len(chunks) == 1 else chunk,
                chat_id=chat_id,
                parse_mode=None,
                reply_markup=reply_markup,
                api_url=api_url,
            )
        if not is_last:
            await asyncio.sleep(0.5)


def detect_options(text: str) -> dict | None:
    """Detect numbered options (1. Option, 2. Option) and create inline keyboard."""
    # Look for patterns like "1.", "2.", "3." at start of lines
    pattern = r"^(\d+)[\.\)]\s+"
    matches = re.findall(pattern, text, re.MULTILINE)

    if not matches or len(matches) < 2:
        return None

    # Get unique numbers, max 8 buttons
    numbers = sorted(set(matches))[:8]

    # Create inline keyboard with number buttons
    buttons = [[{"text": n, "callback_data": f"reply:{n}"} for n in numbers[:4]]]
    if len(numbers) > 4:
        buttons.append([{"text": n, "callback_data": f"reply:{n}"} for n in numbers[4:8]])

    return {"inline_keyboard": buttons}


def split_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks, trying to break at newlines."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks


@app.post("/notify/{event_type}")
async def notify(event_type: str, request: Request):
    """Called by Claude hooks to send notifications."""
    summary = None
    working_dir = None
    try:
        data = await request.json()
        summary = data.get("summary")
        working_dir = data.get("working_dir")
    except Exception:
        pass

    # Determine which bot to notify based on working_dir
    target_bot = bots.get("dev")
    if working_dir and bots.get("gtd"):
        gtd_bot = bots["gtd"]
        if gtd_bot.fixed_working_dir and working_dir.startswith(gtd_bot.fixed_working_dir):
            target_bot = gtd_bot

    if not target_bot:
        return {"ok": False, "error": "No bot configured"}

    if event_type == "completed":
        msg = "✅ <b>Claude has completed the task.</b>"
        if working_dir:
            dir_name = working_dir.split("/")[-1]
            msg = f"✅ <b>Claude has completed</b> (<code>{dir_name}</code>)"
        if summary:
            summary_text = summary[:1500] if len(summary) > 1500 else summary
            try:
                summary_html = markdown_to_telegram_html(summary_text)
            except Exception:
                import html
                summary_html = html.escape(summary_text)
            msg += f"\n\n{summary_html}"
    elif event_type == "waiting":
        msg = "⏸ Claude is waiting for input."
    else:
        msg = f"📢 Claude event: {event_type}"

    await telegram.send_message(msg, chat_id=target_bot.chat_id, parse_mode="HTML", api_url=target_bot.api_url)
    return {"ok": True}


@app.post("/webhook/email")
async def email_webhook(request: Request):
    """Handle email notifications from Google Apps Script."""
    import hmac

    # Verify secret
    secret = request.headers.get("x-webhook-secret", "")
    if not settings.webhook_secret or secret != settings.webhook_secret:
        logger.warning("Unauthorized email webhook request")
        return {"error": "Unauthorized"}

    data = await request.json()

    gtd_bot = bots.get("gtd")
    if not gtd_bot:
        logger.error("Email webhook called but GTD bot not configured")
        return {"error": "GTD bot not configured"}

    # Process asynchronously so Google Apps Script doesn't timeout
    asyncio.create_task(_process_email(data, gtd_bot))

    return {"status": "accepted"}


async def _process_email(data: dict, bot: BotConfig):
    """Process an incoming email via Claude GTD."""
    from_addr = data.get("from", "unknown")
    subject = data.get("subject", "(no subject)")
    body = data.get("body", "")[:2000]
    date = data.get("date", "")

    logger.info(f"Processing email: '{subject}' from {from_addr}")

    prompt = (
        f"📧 **EMAIL +CLAUDE REÇU** - Applique les règles de la section \"9. EMAILS ENTRANTS\" de ton prompt.\n\n"
        f"---\n"
        f"**De** : {from_addr}\n"
        f"**Sujet** : {subject}\n"
        f"**Date** : {date}\n\n"
        f"**Contenu** :\n{body}\n"
        f"---\n\n"
        f"Traite cet email selon tes règles de catégorisation.\n"
        f"IMPORTANT : Réponds aussi à l'expéditeur par email (utilise send_email via Gmail MCP) "
        f"avec un résumé de ce que tu as fait.\n"
        f"NE PAS relire l'email via Gmail, le contenu est ci-dessus."
    )

    if gtd_queue is not None:
        item = QueueItem(
            prompt=prompt,
            source="email",
            chat_id=bot.chat_id,
            new_session=True,
            metadata={"subject": subject, "from": from_addr},
        )
        added = await gtd_queue.enqueue(item)
        if not added:
            await telegram.send_message(
                f"⚠️ Queue pleine, email ignoré: {subject}",
                chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
            )
    else:
        # Fallback: direct execution (shouldn't happen in production)
        try:
            runner = get_runner_for_bot(bot)
            result = await runner.run(
                prompt,
                new_session=True,
                bypass_permissions=True,
                system_prompt=bot.system_prompt,
                mcp_config=bot.mcp_config_path,
            )
            header = f"📧 <b>Email +claude traité</b>\n\nDe: {from_addr}\nSujet: {subject}\n\n"
            await telegram.send_message(header, chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url)
            if result.text:
                await send_response(result.text, bot.chat_id, session_name="gtd", api_url=bot.api_url)
            else:
                await telegram.send_message("(pas de réponse)", chat_id=bot.chat_id, api_url=bot.api_url)
        except Exception as e:
            logger.exception("Email processing error")
            await telegram.send_message(
                f"❌ Erreur traitement email: <code>{e}</code>\nSujet: {subject}",
                chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
            )


@app.post("/cron/{reminder_type}")
async def cron_reminder(reminder_type: str):
    """Handle cron reminders (morning/evening/weekly)."""
    prompt = CRON_PROMPTS.get(reminder_type)
    if not prompt:
        return {"error": f"Unknown reminder type: {reminder_type}"}

    gtd_bot = bots.get("gtd")
    if not gtd_bot:
        return {"error": "GTD bot not configured"}

    # Process asynchronously so curl returns immediately
    asyncio.create_task(_process_cron(prompt, reminder_type, gtd_bot))

    return {"status": "accepted", "type": reminder_type}


async def _process_cron(prompt: str, reminder_type: str, bot: BotConfig):
    """Process a cron reminder via Claude GTD."""
    logger.info(f"Processing cron reminder: {reminder_type}")
    if gtd_queue is not None:
        item = QueueItem(
            prompt=prompt,
            source="cron",
            chat_id=bot.chat_id,
            new_session=True,
        )
        added = await gtd_queue.enqueue(item)
        if not added:
            logger.warning(f"Queue full, skipping cron {reminder_type}")
    else:
        # Fallback: direct execution
        try:
            runner = get_runner_for_bot(bot)
            result = await runner.run(
                prompt,
                new_session=True,
                bypass_permissions=True,
                system_prompt=bot.system_prompt,
                mcp_config=bot.mcp_config_path,
            )
            if result.text:
                await send_response(result.text, bot.chat_id, session_name="gtd", api_url=bot.api_url)
        except Exception as e:
            logger.exception(f"Cron reminder error ({reminder_type})")
            await telegram.send_message(
                f"❌ Erreur rappel {reminder_type}: <code>{e}</code>",
                chat_id=bot.chat_id,
                parse_mode="HTML",
                api_url=bot.api_url,
            )


@app.post("/test")
async def test_message(request: Request):
    """Test endpoint - send a message as if from Telegram."""
    data = await request.json()
    text = data.get("text", "")

    dev_bot = bots.get("dev")
    if not dev_bot:
        return {"error": "No dev bot configured"}

    chat_id = str(dev_bot.chat_id)

    if not text:
        return {"error": "No text provided"}

    if text.startswith("/"):
        await handle_command(text, chat_id, dev_bot)
    else:
        runner = get_runner_for_bot(dev_bot)
        continue_session = runner.is_in_conversation()
        await run_claude(text, chat_id, dev_bot, continue_session=continue_session)

    return {"ok": True, "text": text}


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "claude_telegram.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
