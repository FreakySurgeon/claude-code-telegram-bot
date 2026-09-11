"""FastAPI application - Telegram webhook handler."""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env into os.environ so Claude CLI subprocesses inherit all vars
# (needed for MCP servers that use ${ENV_VAR} references in .mcp.json)
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .bots import BotConfig, create_bots


# Pipeline-enabled crons — run Python script instead of Claude CLI + MCP
# These pipelines pre-assemble context via REST API, then make a single Claude call
PIPELINE_CRONS = {"morning", "evening", "whatsapp", "sent-emails", "gdrive-inbox", "enrichment", "limitless", "omi", "agent-tasks", "garmin-sync", "weekly", "zulip"}

# Per-pipeline timeout overrides (default: 600s)
PIPELINE_TIMEOUTS = {"enrichment": 1800, "agent-tasks": 1800, "garmin-sync": 300, "weekly": 1800, "zulip": 1200}

# Model assignment per cron type — lightweight crons use Haiku
CRON_MODELS: dict[str, str | None] = {
    "whatsapp": "haiku",
    "gdrive-inbox": None,            # Sonnet — needs good judgment for file classification & routing
    "morning": None,          # default (Sonnet)
    "evening": None,          # default (Sonnet)
    "weekly": None,           # default (Sonnet)
    "calendar-actions": None, # default (Sonnet)
    "sent-emails": None,      # Sonnet — needs judgment for matching emails to cards
}

def _load_cron_prompt(reminder_type: str) -> str | None:
    """Load a cron prompt from the configured directory, or return None."""
    from .config import settings
    if not settings.gtd_cron_prompts_dir:
        return None
    prompt_file = Path(settings.gtd_cron_prompts_dir) / f"{reminder_type}.txt"
    try:
        return prompt_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"Cron prompt file not found: {prompt_file}")
        return None
    except Exception as e:
        logger.error(f"Failed to read cron prompt {prompt_file}: {e}")
        return None


def _load_post_session_prompt() -> str | None:
    """Load the post-session memory enrichment prompt, or return None."""
    from .config import settings
    if not settings.gtd_post_session_prompt:
        return None
    try:
        return Path(settings.gtd_post_session_prompt).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"Post-session prompt file not found: {settings.gtd_post_session_prompt}")
        return None
    except Exception as e:
        logger.error(f"Failed to read post-session prompt: {e}")
        return None

from .adapters.telegram import api as telegram  # bot bootstrap only (getMe, webhook)
from .claude import sessions
from .config import settings
from .tunnel import tunnel, CloudflareTunnel
from .queue import QueueItem, RequestQueue, process_queue_item, PersistentQueue, ApiStatus
from .pending_actions import (
    add_action,
    cleanup_actions,
    is_duplicate,
)
from .whatsapp_health import ensure_whatsapp_bridge
from . import state
from .adapters import build_outbounds
from .adapters.telegram import handlers, computer_use
from .conversations import SessionStore, runner_key
from .notifications import NotificationService
from .ports import Action, Event
from .routing import RoutingPolicy, event_type_for, parse_severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# /channels/inject only answers local callers (tests, ops scripts on the host).
INJECT_HOSTS = {"127.0.0.1", "::1"}


def _notifier() -> NotificationService:
    """The NotificationService built at startup (Telegram-only before that)."""
    if state.notifications is not None:
        return state.notifications
    policy = RoutingPolicy.default()
    return NotificationService(policy, build_outbounds(policy, state.bots, settings.resolved_data_dir))


async def _submit_to_gtd_queue(item: QueueItem) -> int | None:
    """Queue position of the item, or None when the GTD queue is full/missing."""
    if state.gtd_queue is None or not await state.gtd_queue.enqueue(item):
        return None
    return state.gtd_queue.size


def _start_zulip_inbound(policy: RoutingPolicy, outbound, data_dir: Path, gtd_bot: BotConfig) -> None:
    """Listen to Zulip through its event queue and route messages to the ConversationService."""
    from .adapters import ZULIP_STATE_FILE
    from .adapters.zulip.inbound import ZulipInbound
    from .conversations import ConversationService

    cfg = policy.channel_config("zulip")
    inbound = ZulipInbound(
        outbound.client,
        state_path=Path(data_dir) / ZULIP_STATE_FILE,
        listen_streams=cfg.get("listen_streams") or [],
        mention_streams=cfg.get("mention_streams") or [],
        dm=bool(cfg.get("dm", True)),
        owned=outbound.owned,
    )
    contexts: dict[str, str] = {}
    context_path = cfg.get("context_prompt")
    if context_path:
        try:
            contexts["zulip"] = Path(context_path).read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"Zulip context prompt unreadable ({context_path}): {e}")
    state.conversations = ConversationService(
        notifications=state.notifications,
        queue_submit=_submit_to_gtd_queue,
        sessions_manager=sessions,
        working_dir=gtd_bot.fixed_working_dir or os.getcwd(),
        session_store=state.session_store,
        data_dir=data_dir,
        ttl_hours=float(cfg.get("session_ttl_hours", 12)),
        channel_contexts=contexts,
        inbound_channels={"zulip": inbound},
    )
    state.zulip_inbound = inbound
    state.inbounds.append(inbound)
    state.inbound_tasks.append(asyncio.create_task(inbound.run(state.conversations.handle)))
    logger.info(f"Zulip inbound started: listen={sorted(inbound.listen_streams)} "
                f"mention={len(inbound.mention_streams)} streams dm={inbound.dm}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and teardown."""
    # Initialize bots
    state.bots = create_bots()
    for bot_name, bot in state.bots.items():
        state.chat_to_bot[str(bot.chat_id)] = bot_name
        # Fetch bot username via getMe
        try:
            me = await telegram.get_me(api_url=bot.api_url)
            bot.username = me.get("result", {}).get("username")
            logger.info(f"Bot {bot_name}: @{bot.username}")
        except Exception as e:
            logger.warning(f"Failed to fetch username for {bot_name}: {e}")
    logger.info(f"Initialized bots: {list(state.bots.keys())}")

    # Channels: routing policy + one outbound per configured channel
    policy = RoutingPolicy.from_file(settings.channel_routing_path, settings.channel_env_file)
    data_dir = settings.resolved_data_dir
    outbounds = build_outbounds(policy, state.bots, data_dir)
    state.notifications = NotificationService(policy, outbounds)
    state.session_store = SessionStore(data_dir / "channel-sessions.json")
    logger.info(f"Channels: default={policy.default_channel} urgent={policy.config.get('urgent')} "
                f"outbounds={sorted(outbounds)}")

    mode = settings.mode

    # Tunnel mode — only for dev bot
    if mode == "tunnel":
        if not CloudflareTunnel.is_available():
            logger.warning("cloudflared not found, falling back to polling mode")
            mode = "polling"
        else:
            logger.info("Starting Cloudflare tunnel...")
            tunnel.port = settings.port
            state.tunnel_url = await tunnel.start()

            if state.tunnel_url:
                webhook_url = f"{state.tunnel_url}{settings.webhook_path}"
                logger.info(f"Tunnel URL: {state.tunnel_url}")
                logger.info(f"Setting webhook: {webhook_url}")
                try:
                    await telegram.set_webhook_with_retry(webhook_url, api_url=state.bots["dev"].api_url)
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
        await telegram.set_webhook(webhook_url, api_url=state.bots["dev"].api_url)

    # Polling mode (fallback or default)
    if mode == "polling":
        logger.info("Starting polling mode...")
        for bot_name, bot in state.bots.items():
            await telegram.delete_webhook(api_url=bot.api_url)
            task = asyncio.create_task(handlers.poll_updates(bot))
            state.polling_tasks.append(task)

    # Start GTD queue worker
    gtd_bot_instance = state.bots.get("gtd")
    if gtd_bot_instance:
        state.gtd_queue = RequestQueue(maxsize=30)
        state.queue_worker_task = asyncio.create_task(queue_worker(state.gtd_queue, gtd_bot_instance))

        # Initialize persistent queue for API unavailability
        import os
        working_dir = gtd_bot_instance.fixed_working_dir or os.getcwd()
        state.persistent_queue = PersistentQueue(Path(working_dir) / "data" / "queue")
        state.api_status = ApiStatus()
        if not state.persistent_queue.is_empty:
            logger.info(f"Found {state.persistent_queue.size} items in persistent queue from previous run")

        if "zulip" in outbounds:
            _start_zulip_inbound(policy, outbounds["zulip"], data_dir, gtd_bot_instance)

    yield

    # Cleanup
    for inbound in state.inbounds:
        inbound.stop()
    for task in state.inbound_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if "zulip" in outbounds:
        await outbounds["zulip"].client.aclose()

    if state.queue_worker_task:
        state.queue_worker_task.cancel()
        try:
            await state.queue_worker_task
        except asyncio.CancelledError:
            pass

    for task in state.polling_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if tunnel.is_running:
        await telegram.delete_webhook(api_url=state.bots["dev"].api_url)
        await tunnel.stop()

    if mode == "webhook" and settings.webhook_url:
        await telegram.delete_webhook(api_url=state.bots["dev"].api_url)



async def _replay_persistent_queue(bot: BotConfig, queue: RequestQueue):
    """Replay all items from the persistent queue after API recovery."""
    items_files = state.persistent_queue.list_items_with_paths()
    if not items_files:
        return

    count = len(items_files)
    logger.info(f"Replaying {count} items from persistent queue")
    await _notifier().publish(Event(
        "llm_provider", "info",
        body=f"✅ Claude est de retour ! Traitement de {count} message(s) en attente...",
    ))

    for item, filepath in items_files:
        added = await queue.enqueue(item)
        if added:
            state.persistent_queue.delete(filepath)
        else:
            logger.warning("Queue full during replay, stopping")
            break


def _runner_for_item(item: QueueItem, bot: BotConfig):
    """Telegram keeps one runner per topic; other channels one per conversation key."""
    ref = item.ref_or_legacy()
    if ref is None or ref.channel == "telegram":
        thread_id = (ref.thread_id if ref else None) or item.thread_id or 0
        return handlers.get_runner(bot, thread_id=thread_id)
    working_dir = bot.fixed_working_dir or sessions.default_dir
    return sessions.get_session(working_dir, thread_id=runner_key(ref))


async def queue_worker(queue: RequestQueue, bot: BotConfig):
    """Worker loop: dequeue and process items one at a time."""
    logger.info("Queue worker started")
    while True:
        try:
            item = await queue.dequeue()
            logger.info(f"Processing queued {item.source} request (retry={item.retry_count})")
            runner = _runner_for_item(item, bot)
            await process_queue_item(item, runner, bot, queue=queue,
                                     persistent_queue=state.persistent_queue,
                                     api_status=state.api_status,
                                     notifications=state.notifications,
                                     session_store=state.session_store)
            # After successful processing, replay persistent queue if API recovered
            if state.persistent_queue and not state.persistent_queue.is_empty and state.api_status and not state.api_status.unavailable:
                await _replay_persistent_queue(bot, queue)
        except asyncio.CancelledError:
            logger.info("Queue worker stopped")
            break
        except Exception:
            logger.exception("Queue worker error")
            await asyncio.sleep(1)


app = FastAPI(title="Claude Telegram", lifespan=lifespan)
app.include_router(computer_use.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "claude_running": sessions.any_running(),
        "active_sessions": sum(len(threads) for threads in sessions.sessions.values()),
        "active_dirs": len(sessions.sessions),
        "queue_size": state.gtd_queue.size if state.gtd_queue else 0,
        "persistent_queue_size": state.persistent_queue.size if state.persistent_queue else 0,
        "api_unavailable": state.api_status.unavailable if state.api_status else False,
    }


@app.post(settings.webhook_path)
async def webhook(request: Request):
    """Handle Telegram webhook updates (dev bot only in tunnel/webhook mode)."""
    data = await request.json()
    logger.info(f"Received update: {data}")

    dev_bot = state.bots.get("dev")
    if not dev_bot:
        return {"ok": False}

    await handlers.handle_update(data, dev_bot)

    return {"ok": True}


@app.post("/notify/{event_type}")
async def notify(event_type: str, request: Request):
    """Called by Claude hooks to send notifications."""
    summary = None
    working_dir = None
    session_id = None
    try:
        data = await request.json()
        summary = data.get("summary")
        working_dir = data.get("working_dir")
        session_id = data.get("session_id")
    except Exception:
        pass

    logger.info(f"notify/{event_type}: working_dir={working_dir}, session_id={session_id}, has_summary={summary is not None}")

    # --- Fitness debrief: enqueue in GTD queue for full skill-based debrief ---
    if event_type == "fitness-completed" and summary:
        gtd_bot = state.bots.get("gtd")
        if gtd_bot and state.gtd_queue is not None:
            # Dedicated conversation for the debrief, on the routed channel
            ref = await _notifier().open_conversation(
                Event("fitness", "normal", title="🏋️ Debrief séance fitness"))
            thread_id = ref.thread_id if ref is not None and ref.channel == "telegram" else None

            prompt = (
                f"🏋️ Séance terminée — déclenche le debrief fitness.\n\n"
                f"Résultats de la séance :\n{summary}\n\n"
                f"Charge le skill fitness-coach (scripts/skills/fitness-coach.txt) "
                f"et exécute le protocole de debrief post-séance complet."
            )
            item = QueueItem(
                prompt=prompt,
                source="fitness",
                chat_id=gtd_bot.chat_id,
                model="sonnet",
                new_session=True,
                timeout=600,
                metadata={"type": "fitness-debrief"},
                thread_id=thread_id,
                conversation=ref,
                event_type="fitness",
            )
            added = await state.gtd_queue.enqueue(item)
            logger.info(f"Fitness debrief enqueued: {added}, conversation={ref.key if ref else None}")
            return {"ok": True, "enqueued": added}
        else:
            logger.warning("Fitness debrief: no GTD bot or queue available")
            return {"ok": False, "error": "No GTD bot/queue"}

    # Always notify via dev bot — hook.py already skips bot-triggered sessions
    # (CLAUDE_TELEGRAM_BOT env check), so /notify only fires for external CLI
    # sessions (VS Code etc.) which should always go to the dev bot.
    target_bot = state.bots.get("dev")

    if not target_bot:
        return {"ok": False, "error": "No bot configured"}

    actions: list[Action] = []

    # Markdown body: each outbound renders it for its own chat
    if event_type == "completed":
        msg = "✅ **Claude has completed the task.**"
        if working_dir:
            dir_name = working_dir.split("/")[-1]
            msg = f"✅ **Claude has completed** (`{dir_name}`)"
        if summary:
            # Truncate to ~5 lines for preview
            lines = summary.split("\n")
            preview = "\n".join(lines[:5])
            if len(lines) > 5:
                preview += "\n…"
            # Cap at 800 chars
            if len(preview) > 800:
                preview = preview[:800] + "…"
            msg += f"\n\n{preview}"
        # Add "Continue" button if session_id is available
        if session_id:
            actions.append(Action("Continue ➜", f"resume:{session_id}"))
            # Store working_dir for the resume callback (can't fit in callback_data)
            if working_dir:
                state.resume_working_dirs[session_id] = working_dir
    elif event_type == "waiting":
        msg = "⏸ Claude is waiting for input."
    else:
        msg = f"📢 Claude event: {event_type}"

    await _notifier().publish(Event(f"dev.{event_type}", "info", body=msg, actions=actions))
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

    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        logger.error("Email webhook called but GTD bot not configured")
        return {"error": "GTD bot not configured"}

    # Process asynchronously so Google Apps Script doesn't timeout
    asyncio.create_task(_process_email(data, gtd_bot))

    return {"status": "accepted"}


@app.post("/webhook/omi")
async def webhook_omi(request: Request):
    """Receive Omi Memory Created webhook."""
    # Validate secret — check both header and query param (Omi may use either)
    # Omi appends ?uid=... to the webhook URL, which can corrupt the secret
    # if it uses ? instead of & (e.g. ?secret=XXX?uid=YYY)
    secret = request.headers.get("x-webhook-secret", "")
    if not secret:
        secret = request.query_params.get("secret", "")
        if secret and "?" in secret:
            secret = secret.split("?")[0]

    if not secret or secret != settings.webhook_secret:
        logger.warning("Omi webhook: unauthorized (secret mismatch)")
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Omi webhook: invalid JSON: {e}")
        return JSONResponse({"error": "invalid json"}, status_code=400)

    memory_id = payload.get("id", "unknown")
    transcript_segments = payload.get("transcript_segments", [])

    if not transcript_segments:
        logger.info(f"Omi webhook: empty transcript (memory_id={memory_id}), skipping")
        return JSONResponse({"status": "ok", "processed": False})

    if payload.get("discarded", False):
        logger.info(f"Omi webhook: discarded memory (memory_id={memory_id}), skipping")
        return JSONResponse({"status": "ok", "processed": False})

    # Save payload to pending directory
    pending_dir = Path(settings.gtd_working_dir) / "data" / "omi-pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_file = pending_dir / f"{memory_id}.json"
    pending_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    logger.info(f"Omi webhook: saved memory {memory_id} ({len(transcript_segments)} segments), triggering pipeline")

    # Trigger pipeline asynchronously (reuses cron pipeline pattern)
    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        logger.error("Omi webhook: GTD bot not configured")
        return JSONResponse({"error": "GTD bot not configured"}, status_code=500)

    asyncio.create_task(_process_pipeline_cron("omi", gtd_bot))

    return JSONResponse({"status": "ok", "processed": True, "memory_id": memory_id})


async def _process_email(data: dict, bot: BotConfig):
    """Process an incoming email via Claude GTD triage."""
    from_addr = data.get("from", "unknown")
    subject = data.get("subject", "(no subject)")
    body = data.get("body", "")[:4000]
    date = data.get("date", "")
    cc = data.get("cc", "")
    attachments = data.get("attachments", [])
    has_draft = data.get("hasDraft", False)
    is_from_thomas = data.get("isFromThomas", False)
    email_message_id = data.get("messageId", "")
    email_thread_id = data.get("threadId", "")
    is_reply = data.get("isReply", False)
    thread_context = data.get("threadContext", "")
    thomas_recipient_type = data.get("thomasRecipientType", "to")

    logger.info(f"Processing email triage: '{subject}' from {from_addr} (fromThomas={is_from_thomas}, hasDraft={has_draft})")

    # Skip self-triage: emails sent by the agent itself (from chauvet.t+claude@gmail.com)
    if "chauvet.t+claude@gmail.com" in from_addr.lower():
        logger.info(f"Skipping self-triage email: '{subject}' (sent by agent)")
        return

    # Skip GitHub notifications (defense-in-depth)
    if "notifications@github.com" in from_addr.lower():
        logger.info(f"Skipping GitHub notification: '{subject}'")
        return

    # Skip known automated notifications (defense-in-depth, primary filter is in Apps Script)
    import re
    IGNORED_SUBJECT_PATTERNS = [
        re.compile(r"documents?\s+(patients?\s+)?re[çc]us?", re.IGNORECASE),  # Lifen DMP CMC
        re.compile(r"dmp\s+cmc", re.IGNORECASE),
        re.compile(r"lifen", re.IGNORECASE),
        re.compile(r"nouveau message s[eé]curis[eé] re[çc]u sur mailiz", re.IGNORECASE),  # Mailiz
    ]
    IGNORED_SENDER_PATTERNS = [
        re.compile(r"healthchecks\.io", re.IGNORECASE),  # Revicare monitoring
    ]
    if any(p.search(subject) for p in IGNORED_SUBJECT_PATTERNS):
        logger.info(f"Skipping ignored subject: '{subject}' (auto-notification filter)")
        return
    if any(p.search(from_addr) for p in IGNORED_SENDER_PATTERNS):
        logger.info(f"Skipping ignored sender: '{from_addr}' (auto-notification filter)")
        return

    # Topic created lazily in queue — only when there's output to send (not for Claude/Info)
    thread_id = None

    # Build attachment info
    attachment_info = ""
    if attachments:
        att_lines = []
        for att in attachments:
            att_lines.append(f"  - {att.get('name', '?')} ({att.get('mimeType', '?')}, {att.get('size', 0)} bytes)")
        attachment_info = f"\n**Pièces jointes** :\n" + "\n".join(att_lines) + "\n"

    # Build draft info
    draft_info = ""
    if has_draft:
        draft_info = "\n**⚠️ Un brouillon de réponse existe déjà dans ce thread** (probablement Jace). Lis-le via Gmail MCP avant de décider si tu dois en créer un autre.\n"

    # Build CC context info
    cc_context = ""
    if thomas_recipient_type == "cc":
        cc_context = (
            "\n**📋 THOMAS EST EN COPIE (CC)** — Thomas n'est PAS le destinataire principal de cet email. "
            "Il est en copie pour information. Adapte ton analyse en conséquence :\n"
            "- Par défaut, cet email est **informatif** pour Thomas (Claude/Info)\n"
            "- Ne lui attribue PAS d'action sauf si le contenu le mentionne explicitement ou lui demande quelque chose\n"
            "- Si un tiers confirme une action (paiement, réponse, validation), **vérifie si une carte Trello existe** pour cette action et marque-la comme terminée\n"
            "- Note les infos utiles dans `faits-recents.md` (ex: Flora a payé X, un collègue a confirmé Y)\n"
        )
    elif thomas_recipient_type == "none":
        cc_context = (
            "\n**⚠️ THOMAS N'EST NI EN TO: NI EN CC:** — Cet email est probablement arrivé via un forward ou une liste. "
            "Traite-le comme informatif sauf preuve du contraire.\n"
        )

    # Build reply context info
    reply_info = ""
    if is_reply:
        reply_info = (
            "\n**🔄 RÉPONSE DANS UN THREAD DÉJÀ TRIÉ** — Ceci est une nouvelle réponse dans une conversation existante. "
            "Le thread avait déjà été traité mais un nouveau message est arrivé. "
            "Tu dois re-évaluer la situation : créer/mettre à jour la carte Trello, préparer un brouillon de réponse, "
            "créer un événement Calendar si pertinent.\n"
        )
        if thread_context:
            reply_info += f"\n**Contexte du thread (messages précédents)** :\n{thread_context}\n"

    prompt = (
        f"📧 **TRIAGE EMAIL{'  — RÉPONSE' if is_reply else ''}** - Applique les règles de la section \"Triage Email\" de ton prompt.\n\n"
        f"---\n"
        f"**De** : {from_addr}\n"
        f"**À** : {data.get('to', '')}\n"
        f"**CC** : {cc}\n"
        f"**Sujet** : {subject}\n"
        f"**Date** : {date}\n"
        f"**Message ID** : {email_message_id}\n"
        f"**Thread ID** : {email_thread_id}\n"
        f"**Email de Thomas** : {'OUI' if is_from_thomas else 'NON'}\n"
        f"**Position Thomas** : {'Destinataire principal (To:)' if thomas_recipient_type == 'to' else 'En copie (CC:)' if thomas_recipient_type == 'cc' else 'Ni To: ni CC:'}\n"
        f"{cc_context}"
        f"{attachment_info}"
        f"{draft_info}"
        f"{reply_info}\n"
        f"**Contenu** :\n{body}\n"
        f"---\n\n"
        f"Traite cet email selon les règles de triage.\n"
        f"NE PAS relire l'email via Gmail, le contenu est ci-dessus.\n"
        f"Tu peux utiliser Gmail MCP pour : chercher dans le thread, lire les brouillons, "
        f"télécharger les pièces jointes, appliquer les labels.\n"
        f"⚠️ Pour envoyer le résumé, utilise UNIQUEMENT `scripts/send-agent-email.py` (SMTP agent@freakymex.ovh) "
        f"avec --gmail-id \"{email_message_id}\" pour le threading. "
        f"INTERDIT d'utiliser `send_email` ou `reply` du MCP Gmail pour les résumés.\n\n"
        f"⚠️ RÈGLE CRITIQUE : Si tu classifies cet email comme `Claude/Info` (newsletter, notification, "
        f"promo, spam, confirmation de commande, notification calendrier, etc.), tu dois UNIQUEMENT "
        f"appliquer le label Gmail `Claude/Info` via modify_email. INTERDICTION ABSOLUE d'appeler "
        f"`send_email` ou `reply` pour les emails classés Info. Zéro email de résumé. "
        f"Juste le label, puis termine."
    )

    if state.gtd_queue is not None:
        item = QueueItem(
            prompt=prompt,
            source="email",
            chat_id=bot.chat_id,
            model="sonnet",
            new_session=True,
            timeout=900,  # 15 min for email (MCP-heavy: Gmail + Trello + GDrive)
            metadata={"subject": subject, "from": from_addr},
            thread_id=thread_id,
            event_type="email_triage",
        )
        added = await state.gtd_queue.enqueue(item)
        if not added:
            await _notifier().publish(Event(
                "email_triage", "normal", title=f"Email: {subject[:60]}",
                body=f"⚠️ Queue pleine, email ignoré : {subject}",
            ))
    else:
        # Fallback: direct execution (shouldn't happen in production)
        title = f"Email: {subject[:60]}"
        try:
            runner = handlers.get_runner(bot, thread_id=0)
            result = await runner.run(
                prompt,
                model="haiku",
                new_session=True,
                bypass_permissions=True,
                system_prompt=bot.system_prompt,
                mcp_config=bot.mcp_config_path,
            )
            await _notifier().publish(Event("email_triage", "normal", title=title,
                                            body=result.text or "(pas de réponse)"))
        except Exception as e:
            logger.exception("Email processing error")
            await _notifier().publish(Event("email_triage", "normal", title=title,
                                            body=f"❌ Erreur traitement email : `{e}`\nSujet : {subject}"))


ZULIP_FALLBACK_DELAY = 120  # seconds the event queue gets before the webhook pipeline steps in


def _zulip_pending_dir() -> Path:
    return Path(settings.gtd_working_dir) / "data" / "zulip-pending"


def _purge_seen_zulip_payloads(inbound) -> int:
    """Drop webhook payloads the Zulip event queue already answered."""
    removed = 0
    for path in _zulip_pending_dir().glob("*.json"):
        if inbound.seen(path.stem):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info(f"Zulip: purged {removed} webhook payload(s) already handled by the event queue")
    return removed


async def _delayed_zulip_fallback(message_id, bot: BotConfig, delay: float = ZULIP_FALLBACK_DELAY):
    """Run the webhook pipeline only if the event queue did not pick the message up."""
    await asyncio.sleep(delay)
    inbound = state.zulip_inbound
    if inbound is not None and inbound.seen(message_id):
        (_zulip_pending_dir() / f"{message_id}.json").unlink(missing_ok=True)
        return
    if inbound is not None:
        try:
            inbound.mark_seen(message_id)  # the event queue must not answer it a second time
            inbound.save_state()
        except (TypeError, ValueError):
            pass
    logger.warning(f"Zulip: message {message_id} not seen by the event queue after {delay}s, running pipeline")
    await _process_pipeline_cron("zulip", bot)


@app.post("/webhook/zulip")
async def webhook_zulip(request: Request):
    """Receive a Zulip outgoing-webhook bot mention or DM.

    Zulip times out quickly, so we only persist the payload and return an
    empty body (which tells Zulip to post nothing). The agent's real answer
    is posted back into the thread by scripts/pipelines/zulip.py — or, when the
    Zulip event queue runs, by the ConversationService (the webhook is then
    only a fallback for messages the event queue missed).
    """
    import hmac

    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Zulip webhook: invalid JSON: {e}")
        return JSONResponse({"error": "invalid json"}, status_code=400)

    # Zulip authenticates itself with the bot service token in the body.
    expected = os.environ.get("ZULIP_WEBHOOK_TOKEN", "")
    got = payload.get("token", "")
    if not expected or not hmac.compare_digest(str(got), expected):
        logger.warning("Zulip webhook: unauthorized (token mismatch)")
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    message = payload.get("message", {}) or {}
    message_id = message.get("id", "unknown")

    if not (payload.get("data") or message.get("content", "")).strip():
        logger.info(f"Zulip webhook: empty message {message_id}, skipping")
        return JSONResponse({})

    inbound = state.zulip_inbound
    if inbound is not None and inbound.seen(message_id):
        logger.info(f"Zulip webhook: message {message_id} already handled by the event queue")
        return JSONResponse({})

    pending_dir = _zulip_pending_dir()
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{message_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    logger.info(f"Zulip webhook: queued message {message_id} from {message.get('sender_email','?')}")

    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        logger.error("Zulip webhook: GTD bot not configured")
        return JSONResponse({"error": "GTD bot not configured"}, status_code=500)

    if inbound is not None:
        # The event queue normally answers first; the pipeline only catches what it missed.
        asyncio.create_task(_delayed_zulip_fallback(message_id, gtd_bot))
    else:
        asyncio.create_task(_process_pipeline_cron("zulip", gtd_bot))

    # Empty object = "bot has nothing to say right now".
    return JSONResponse({})


@app.post("/channels/inject")
async def channels_inject(request: Request):
    """Local test hook: publish an Event or feed an InboundMessage to the core.

    ``{"event": {"type", "severity", "title", "body"}}`` goes through the
    RoutingPolicy like any cron output; ``{"inbound": {"channel",
    "conversation_id", "topic", "text", "user", "user_name", "message_id"}}``
    goes through the ConversationService like a real chat message.
    Requires ``X-Webhook-Secret`` and a loopback caller.
    """
    import hmac

    secret = request.headers.get("x-webhook-secret", "")
    if not settings.webhook_secret or not hmac.compare_digest(secret, settings.webhook_secret):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    host = request.client.host if request.client else ""
    if host not in INJECT_HOSTS:
        logger.warning(f"channels/inject refused for remote host {host}")
        return JSONResponse({"error": "forbidden"}, status_code=403)
    data = await request.json()

    if "event" in data:
        raw = data["event"] or {}
        event = Event(
            raw.get("type", "test"), raw.get("severity", "info"),
            title=raw.get("title", ""), body=raw.get("body", ""),
        )
        deliveries = await _notifier().publish(event)
        return {"ok": True, "deliveries": [
            {"channel": d.channel, "conversation": d.ref.key if d.ref else None,
             "message_ids": d.message_ids, "error": d.error}
            for d in deliveries or []
        ]}

    if "inbound" in data:
        if state.conversations is None:
            return JSONResponse({"error": "no conversation service"}, status_code=409)
        raw = data["inbound"] or {}
        from .ports import ConversationRef, InboundMessage

        channel = raw.get("channel", "zulip")
        ref = ConversationRef(channel, raw["conversation_id"], topic=raw.get("topic"))
        await state.conversations.handle(InboundMessage(
            channel=channel, conversation_id=ref.conversation_id, user=raw.get("user", ""),
            text=raw.get("text", ""), reply_to=ref, message_id=raw.get("message_id"),
            user_name=raw.get("user_name", ""),
        ))
        return {"ok": True, "conversation": ref.key}

    return JSONResponse({"error": "expected 'event' or 'inbound'"}, status_code=400)


@app.post("/cron/calendar-actions")
async def cron_calendar_actions():
    """Scan tomorrow's calendar for <agent> prompts and execute them."""
    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        return {"error": "GTD bot not configured"}

    asyncio.create_task(_process_calendar_actions(gtd_bot))
    return {"status": "accepted", "type": "calendar-actions"}


async def _process_calendar_actions(bot: BotConfig):
    """Two-phase calendar action processing."""
    import json as json_mod
    from pathlib import Path
    from datetime import datetime

    working_dir = bot.fixed_working_dir or sessions.default_dir
    pending_path = Path(working_dir) / "data" / "pending-actions.json"
    scan_path = Path(working_dir) / "data" / "calendar-scan.json"

    # Cleanup old actions first
    cleanup_actions(pending_path)

    # --- Phase 1: Scanner session ---
    logger.info("Calendar actions: Phase 1 — scanning tomorrow's events")

    scan_prompt = _load_cron_prompt("calendar-scan")
    if not scan_prompt:
        logger.error("Calendar actions: calendar-scan.txt not found")
        return

    # Run scanner synchronously (wait for completion)
    runner = handlers.get_runner(bot, thread_id=0)
    try:
        result = await runner.run(
            scan_prompt,
            new_session=True,
            bypass_permissions=True,
            system_prompt=bot.system_prompt,
            mcp_config=bot.mcp_config_path,
            timeout=120,
        )
        logger.info(f"Calendar actions: scan complete (session {runner.session_id})")
    except Exception as e:
        logger.error(f"Calendar actions: scan failed: {e}")
        return

    # --- Phase 2: Parse results and enqueue actions ---
    logger.info("Calendar actions: Phase 2 — orchestrating action topics")

    try:
        scan_data = json_mod.loads(scan_path.read_text())
    except (FileNotFoundError, json_mod.JSONDecodeError) as e:
        logger.error(f"Calendar actions: failed to read scan results: {e}")
        return

    events = scan_data.get("events", [])
    if not events:
        logger.info("Calendar actions: no events with <agent> tags found")
        return

    # Load action template
    action_template = _load_cron_prompt("calendar-action")
    if not action_template:
        logger.error("Calendar actions: calendar-action.txt template not found")
        return

    enqueued = 0
    for event in events:
        for i, agent in enumerate(event.get("agent_prompts", [])):
            event_id = event.get("event_id", "unknown")
            action_id = f"evt_{event_id}_{i}"
            prompt_text = agent.get("prompt", "")
            confirm = agent.get("confirm", True)

            # Deduplication
            if is_duplicate(pending_path, event_id, prompt_text):
                logger.info(f"Calendar actions: skipping duplicate {action_id}")
                continue

            # Build confirm instructions
            if confirm:
                confirm_instructions = (
                    "Exécute l'instruction ci-dessus. Présente le résultat, puis demande "
                    "confirmation à Thomas :\n"
                    '"✅ Confirmer / ✏️ Modifier / ❌ Annuler"\n\n'
                    "Quand Thomas confirme ou modifie, exécute l'action finale puis mets "
                    "à jour data/pending-actions.json : change le status de l'action "
                    f'(id: "{action_id}") à "confirmed" ou "cancelled".'
                )
            else:
                confirm_instructions = (
                    "Exécute l'instruction ci-dessus directement, sans attendre de "
                    "confirmation. Mets à jour data/pending-actions.json : change le status "
                    f'de l\'action (id: "{action_id}") à "confirmed".'
                )

            # Build prompt from template
            prompt = action_template.format(
                event_title=event.get("title", ""),
                event_date=event.get("date", ""),
                start_time=event.get("start_time", ""),
                end_time=event.get("end_time", ""),
                event_description=event.get("description", ""),
                agent_prompt=prompt_text,
                confirm_instructions=confirm_instructions,
            )

            # Dedicated conversation on the routed channel
            ref = await _notifier().open_conversation(Event(
                "calendar_action", "normal", title=f"📅 {event.get('title', 'Action calendrier')}"))
            if ref is None:
                logger.warning(f"Calendar actions: no conversation opened for {action_id}")
            thread_id = ref.thread_id if ref is not None and ref.channel == "telegram" else None

            # Save to pending-actions.json
            action_entry = {
                "id": action_id,
                "event_id": event_id,
                "event_title": event.get("title", ""),
                "event_date": event.get("date", ""),
                "prompt": prompt_text,
                "confirm": confirm,
                "status": "pending",
                "thread_id": thread_id,
                "conversation": ref.to_dict() if ref is not None else None,
                "created_at": datetime.now().isoformat(),
                "executed_at": None,
                "resolved_at": None,
            }
            add_action(pending_path, action_entry)

            # Enqueue for Claude execution
            if state.gtd_queue is not None:
                item = QueueItem(
                    prompt=prompt,
                    source="cron",
                    chat_id=bot.chat_id,
                    new_session=True,
                    timeout=900,  # 15 min for calendar actions
                    metadata={
                        "reminder_type": "calendar-action",
                        "action_id": action_id,
                    },
                    thread_id=thread_id,
                    conversation=ref,
                    event_type="calendar_action",
                )
                added = await state.gtd_queue.enqueue(item)
                if added:
                    enqueued += 1
                    logger.info(f"Calendar actions: enqueued {action_id} → {ref.key if ref else 'no conversation'}")
            else:
                logger.warning("Calendar actions: queue unavailable, skipping execution")

    logger.info(f"Calendar actions: {enqueued} actions enqueued from {len(events)} events")


@app.post("/cron/{reminder_type}")
async def cron_reminder(reminder_type: str):
    """Handle cron reminders (morning/evening/weekly)."""
    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        return {"error": "GTD bot not configured"}

    # Pipeline-enabled crons bypass Claude CLI + MCP entirely
    if reminder_type in PIPELINE_CRONS:
        if reminder_type == "zulip" and state.zulip_inbound is not None:
            _purge_seen_zulip_payloads(state.zulip_inbound)
        asyncio.create_task(_process_pipeline_cron(reminder_type, gtd_bot))
        return {"status": "accepted", "type": reminder_type, "mode": "pipeline"}

    # Non-pipeline crons need a prompt file
    prompt = _load_cron_prompt(reminder_type)
    if not prompt:
        return {"error": f"Unknown reminder type: {reminder_type}"}

    # Process asynchronously so curl returns immediately
    asyncio.create_task(_process_cron(prompt, reminder_type, gtd_bot))

    return {"status": "accepted", "type": reminder_type}


async def _process_pipeline_cron(reminder_type: str, bot: BotConfig):
    """Run a Python pipeline instead of Claude CLI + MCP.

    Pipelines pre-assemble context via REST API calls, then make a single
    Claude CLI call with --tools "" (no MCP). This reduces token usage by
    ~60-70% and eliminates MCP zombie processes.
    """
    import subprocess as sp
    from .config import settings

    logger.info(f"Processing pipeline cron: {reminder_type}")

    # The conversation is opened by the NotificationService only when there is output
    event_type = event_type_for(reminder_type)
    title = f"Cron: {reminder_type}"

    async def _publish(severity: str, body: str) -> None:
        await _notifier().publish(Event(event_type, severity, title=title, body=body))

    working_dir = settings.gtd_working_dir or "."
    pipeline_timeout = PIPELINE_TIMEOUTS.get(reminder_type, 600)
    try:
        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: sp.run(
                ["python3", f"{working_dir}/scripts/run_pipeline.py", reminder_type],
                capture_output=True, text=True, timeout=pipeline_timeout,
                cwd=working_dir,
            ),
        )
        output = proc.stdout.strip()
        severity = "normal"
        if proc.returncode != 0:
            error_msg = proc.stderr.strip() if proc.stderr else "Unknown error"
            logger.error(f"Pipeline {reminder_type} failed (rc={proc.returncode}): {error_msg[:2000]}")
            output = f"❌ Pipeline {reminder_type} error:\n```\n{error_msg[:1500]}\n```"
        else:
            output, severity = parse_severity(output)

        if output and output.upper() != "OK":
            await _publish(severity, output)
        else:
            logger.info(f"Pipeline {reminder_type} completed silently")
    except sp.TimeoutExpired:
        logger.error(f"Pipeline {reminder_type} timed out ({pipeline_timeout}s)")
        # Clean up stale lock files left by killed subprocesses
        lock_file = Path(working_dir) / "data" / f".{reminder_type.replace('-', '_')}.lock"
        if lock_file.exists():
            lock_file.unlink(missing_ok=True)
            logger.info(f"Cleaned up stale lock: {lock_file}")
        # Also check the enrichment lock specifically
        enrichment_lock = Path(working_dir) / "data" / ".enrichment.lock"
        if reminder_type == "enrichment" and enrichment_lock.exists():
            enrichment_lock.unlink(missing_ok=True)
            logger.info("Cleaned up enrichment lock after timeout")
        await _publish("urgent", f"❌ Pipeline {reminder_type} timeout ({pipeline_timeout}s)")
    except Exception as e:
        logger.exception(f"Pipeline {reminder_type} error")
        # Clean up lock files on error too
        for lock_name in [f".{reminder_type.replace('-', '_')}.lock", ".enrichment.lock"]:
            lock_file = Path(working_dir) / "data" / lock_name
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
                logger.info(f"Cleaned up lock after error: {lock_file}")
        await _publish("normal", f"❌ Pipeline {reminder_type} error: `{e}`")


async def _process_cron(prompt: str, reminder_type: str, bot: BotConfig):
    """Process a cron reminder via Claude GTD."""
    logger.info(f"Processing cron reminder: {reminder_type}")

    # Silent crons don't open a conversation (the queue publishes an Event if needed)
    silent = reminder_type in ("whatsapp", "gdrive-inbox", "sent-emails")
    event_type = event_type_for(reminder_type)
    title = f"Cron: {reminder_type}"
    ref = None

    # WhatsApp bridge pre-flight check
    if reminder_type == "whatsapp":
        async def _notify_bridge_down(text: str) -> None:
            await _notifier().publish(Event("whatsapp_bridge", "urgent", body=text))

        bridge_ok = await ensure_whatsapp_bridge(bot.chat_id, bot.api_url, notify=_notify_bridge_down)
        if not bridge_ok:
            logger.warning("WhatsApp bridge down, skipping scan")
            return

    if not silent:
        ref = await _notifier().open_conversation(Event(event_type, "normal", title=title))
        if ref is None:
            logger.warning(f"No conversation opened for cron {reminder_type}")
    thread_id = ref.thread_id if ref is not None and ref.channel == "telegram" else None

    if state.gtd_queue is not None:
        item = QueueItem(
            prompt=prompt,
            source="cron",
            chat_id=bot.chat_id,
            model=CRON_MODELS.get(reminder_type),
            new_session=True,
            timeout=1800,  # 30 min for cron (enrichissement Trello par subagents)
            metadata={"reminder_type": reminder_type},
            thread_id=thread_id,
            conversation=ref,
            event_type=event_type,
        )
        added = await state.gtd_queue.enqueue(item)
        if not added:
            logger.warning(f"Queue full, skipping cron {reminder_type}")
    else:
        # Fallback: direct execution
        try:
            runner = handlers.get_runner(bot, thread_id=thread_id or 0)
            result = await runner.run(
                prompt,
                model=CRON_MODELS.get(reminder_type),
                new_session=True,
                bypass_permissions=True,
                system_prompt=bot.system_prompt,
                mcp_config=bot.mcp_config_path,
            )
            if result.text:
                await _notifier().publish(Event(event_type, "normal", title=title, body=result.text,
                                                conversation_hint=ref))
        except Exception as e:
            logger.exception(f"Cron reminder error ({reminder_type})")
            await _notifier().publish(Event(event_type, "normal", title=title, conversation_hint=ref,
                                            body=f"❌ Erreur rappel {reminder_type} : `{e}`"))


@app.post("/test")
async def test_message(request: Request):
    """Test endpoint - send a message as if from Telegram."""
    data = await request.json()
    text = data.get("text", "")

    dev_bot = state.bots.get("dev")
    if not dev_bot:
        return {"error": "No dev bot configured"}

    chat_id = str(dev_bot.chat_id)

    if not text:
        return {"error": "No text provided"}

    if text.startswith("/"):
        await handlers.handle_command(text, chat_id, dev_bot)
    else:
        runner = handlers.get_runner_for_bot(dev_bot)
        continue_session = runner.is_in_conversation()
        await handlers.run_claude(text, chat_id, dev_bot, continue_session=continue_session)

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
