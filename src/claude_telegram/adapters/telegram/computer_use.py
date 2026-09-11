"""Computer Use (PCFIXE remote browser automation) — moved from main.py.

Telegram UI for launching/confirming/cancelling computer-use jobs.
"""

import asyncio
import html
import logging
from pathlib import Path

from fastapi import APIRouter, Request

from ...bots import BotConfig
from ...topic import generate_provisional_name
from ... import state
from . import api as telegram

logger = logging.getLogger(__name__)

router = APIRouter()


async def _send_computer_use_result(cu_result, provider, buttons: dict,
                                    chat_id: str, bot, thread_id: int,
                                    hint: str = "Réponds ici ou utilise les boutons."):
    """Send computer use result with screenshot photo + buttons."""
    loop = asyncio.get_event_loop()
    job_id = cu_result.metadata.get("job_id", "")
    downloads_dir = cu_result.metadata.get("downloads_dir")

    screenshot_path = await loop.run_in_executor(
        None, lambda: provider.retrieve_screenshot(job_id, downloads_dir),
    )

    status = "✅" if cu_result.success else "❌"
    caption = (
        f"{status} <b>Résultat</b>\n\n{html.escape(cu_result.text[:800])}\n\n"
        f"💬 <i>{hint}</i>"
    )

    if screenshot_path:
        try:
            await telegram.send_photo(
                screenshot_path,
                chat_id=chat_id, caption=caption, parse_mode="HTML",
                api_url=bot.api_url, message_thread_id=thread_id,
                reply_markup=buttons,
            )
        except Exception as e:
            logger.warning(f"Failed to send screenshot: {e}")
            await telegram.send_message(
                caption, chat_id=chat_id, parse_mode="HTML",
                api_url=bot.api_url, message_thread_id=thread_id,
                reply_markup=buttons,
            )
        finally:
            Path(screenshot_path).unlink(missing_ok=True)
    else:
        await telegram.send_message(
            caption, chat_id=chat_id, parse_mode="HTML",
            api_url=bot.api_url, message_thread_id=thread_id,
            reply_markup=buttons,
        )


def _find_computer_use_by_thread(thread_id: int) -> dict | None:
    """Find a pending computer-use job by its Telegram thread_id."""
    for job_id, job in state.pending_computer_use.items():
        if job.get("thread_id") == thread_id:
            return {"job_id": job_id, **job}
    return None


async def _handle_computer_use_text(cu_job: dict, text: str,
                                    chat_id: str, bot: BotConfig, thread_id: int):
    """Handle a free-text message in a computer-use topic — resume with instructions."""
    import sys
    from ...config import settings
    sys.path.insert(0, str(Path(settings.gtd_working_dir or ".")))
    from scripts.providers.computer_use import ComputerUseProvider

    job_id = cu_job["job_id"]
    session_id = cu_job["session_id"]
    destination = cu_job.get("destination")

    await telegram.send_message(
        f"⏳ Envoi de tes instructions au Claude distant...",
        chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
        message_thread_id=thread_id,
    )

    provider = ComputerUseProvider(cu_job["provider_config"])
    loop = asyncio.get_event_loop()

    cu_result = await loop.run_in_executor(
        None,
        lambda: provider.resume(session_id, text),
    )

    if not cu_result.success and cu_result.metadata.get("session_expired"):
        # Fallback to new run
        cu_result = await loop.run_in_executor(
            None,
            lambda: provider.run(f"Contexte : {cu_job['task']}\n\nNouvelle instruction : {text}"),
        )

    new_job_id = cu_result.metadata.get("job_id", "")

    # Retrieve files if any
    if cu_result.downloaded_files and destination:
        retrieved = await loop.run_in_executor(
            None,
            lambda: provider.retrieve_files(
                new_job_id, destination,
                cu_result.metadata.get("downloads_dir"),
            ),
        )
        if retrieved:
            files_msg = "\n".join(f"• <code>{html.escape(f)}</code>" for f in retrieved)
            await telegram.send_message(
                f"📁 <b>Fichiers récupérés</b>\n\n{files_msg}",
                chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
                message_thread_id=thread_id,
            )

    # Update state
    state.pending_computer_use[new_job_id] = {
        "session_id": cu_result.session_id,
        "thread_id": thread_id,
        "provider_config": cu_job["provider_config"],
        "destination": destination,
        "task": cu_job["task"],
    }
    if job_id != new_job_id:
        state.pending_computer_use.pop(job_id, None)

    buttons = {
        "inline_keyboard": [[
            {"text": "✅ Continuer", "callback_data": f"cu:confirm:{new_job_id}"},
            {"text": "✅ Terminé", "callback_data": f"cu:done:{new_job_id}"},
            {"text": "❌ Annuler", "callback_data": f"cu:cancel:{new_job_id}"},
        ]]
    }

    await _send_computer_use_result(
        cu_result, provider, buttons,
        chat_id, bot, thread_id,
        hint="Écris ici pour donner d'autres instructions.",
    )

@router.post("/computer-use")
async def computer_use_endpoint(request: Request):
    """Launch a computer use task on PCFIXE Windows.

    Body: {"task": "...", "destination": "~/gdrive/...", "model": "sonnet", "timeout": 300}

    Creates a Telegram topic, runs the task, sends result with confirmation
    buttons if needed, and handles file retrieval.
    """
    gtd_bot = state.bots.get("gtd")
    if not gtd_bot:
        return {"error": "GTD bot not configured"}

    data = await request.json()
    task = data.get("task", "")
    destination = data.get("destination")
    model = data.get("model", "sonnet")
    timeout = data.get("timeout", 300)

    if not task:
        return {"error": "No task provided"}

    asyncio.create_task(_process_computer_use(task, destination, model, timeout, gtd_bot))
    return {"status": "accepted", "task": task[:100]}


async def _process_computer_use(task: str, destination: str | None,
                                model: str, timeout: int, bot: BotConfig):
    """Run a computer use task with Telegram topic and confirmation buttons."""
    from ...config import settings
    import sys
    sys.path.insert(0, str(Path(settings.gtd_working_dir or ".")))
    from scripts.providers.computer_use import ComputerUseProvider

    provider_config = {
        "computer_use": {
            "ssh_host": "pcfixe",
            "cli_path": "claude",
            "model": model,
            "timeout": timeout,
            "output_dir": r"C:\Users\chauv\computer-use-jobs",
        }
    }
    provider = ComputerUseProvider(provider_config)

    # Health check
    if not provider.health_check():
        await telegram.send_message(
            "❌ <b>Computer Use</b> — PCFIXE is unreachable (éteint ou SSH down)",
            chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
        )
        return

    # Create dedicated topic
    name = generate_provisional_name("🖥️ Computer Use", is_agent=True)
    try:
        result = await telegram.create_forum_topic(bot.chat_id, name, api_url=bot.api_url)
        thread_id = result["result"]["message_thread_id"]
    except Exception as e:
        logger.error(f"Failed to create topic for computer use: {e}")
        return

    # Notify start
    await telegram.send_message(
        f"🖥️ <b>Computer Use — PCFIXE</b>\n\n"
        f"<i>{html.escape(task[:500])}</i>\n\n"
        f"⏳ Exécution en cours...",
        chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
        message_thread_id=thread_id,
    )

    # Run the task (blocking — in executor to not block event loop)
    loop = asyncio.get_event_loop()
    cu_result = await loop.run_in_executor(None, lambda: provider.run(task, model=model, timeout=timeout))

    if not cu_result.success:
        await telegram.send_message(
            f"❌ <b>Échec</b>\n\n<code>{html.escape(cu_result.text[:1500])}</code>",
            chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )
        return

    job_id = cu_result.metadata.get("job_id", "")

    # Check if there are downloaded files to retrieve
    if cu_result.downloaded_files and destination:
        retrieved = await loop.run_in_executor(
            None,
            lambda: provider.retrieve_files(
                job_id, destination,
                cu_result.metadata.get("downloads_dir"),
            ),
        )
        files_msg = "\n".join(f"• <code>{html.escape(f)}</code>" for f in retrieved)
        await telegram.send_message(
            f"📁 <b>Fichiers récupérés</b>\n\n{files_msg}" if retrieved
            else "📁 Aucun fichier à récupérer",
            chat_id=bot.chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )

    # Send result with confirmation buttons
    # Store job state for potential resume
    state.pending_computer_use[job_id] = {
        "session_id": cu_result.session_id,
        "thread_id": thread_id,
        "provider_config": provider_config,
        "destination": destination,
        "task": task,
    }

    buttons = {
        "inline_keyboard": [[
            {"text": "✅ Continuer", "callback_data": f"cu:confirm:{job_id}"},
            {"text": "❌ Annuler", "callback_data": f"cu:cancel:{job_id}"},
        ]]
    }

    await _send_computer_use_result(
        cu_result, provider, buttons,
        bot.chat_id, bot, thread_id,
        hint="Réponds dans ce topic pour donner des instructions.",
    )


async def _handle_computer_use_callback(action: str, job_id: str,
                                        chat_id: str, bot: BotConfig,
                                        callback: dict):
    """Handle computer-use confirmation/cancellation callbacks."""
    pending = state.pending_computer_use.get(job_id)
    thread_id = callback["message"].get("message_thread_id", 0)
    msg_id = callback["message"]["message_id"]

    if not pending:
        await telegram.send_message(
            "⚠️ Session expirée ou déjà traitée.",
            chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )
        return

    # Remove buttons from original message
    try:
        original_text = callback["message"].get("text", "")
        await telegram.edit_message(
            msg_id, original_text,
            chat_id=chat_id, parse_mode=None, api_url=bot.api_url,
            reply_markup={"inline_keyboard": []},
        )
    except Exception:
        pass

    if action == "cancel":
        del state.pending_computer_use[job_id]
        await telegram.send_message(
            "❌ Annulé.",
            chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )
        return

    if action == "confirm":
        await telegram.send_message(
            "✅ Confirmation reçue — reprise de la session...",
            chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )

        import sys
        from ...config import settings
        sys.path.insert(0, str(Path(settings.gtd_working_dir or ".")))
        from scripts.providers.computer_use import ComputerUseProvider

        provider = ComputerUseProvider(pending["provider_config"])
        session_id = pending["session_id"]
        destination = pending.get("destination")

        loop = asyncio.get_event_loop()

        # Try resume, fallback to new run
        cu_result = await loop.run_in_executor(
            None,
            lambda: provider.resume(session_id, "L'utilisateur a confirmé. Continue la tâche."),
        )

        if not cu_result.success and cu_result.metadata.get("session_expired"):
            # Fallback: new run with context
            fallback_task = (
                f"Reprends cette tâche (la session précédente a expiré) : {pending['task']}\n"
                f"L'utilisateur a confirmé, tu peux continuer."
            )
            cu_result = await loop.run_in_executor(
                None,
                lambda: provider.run(fallback_task),
            )

        # Retrieve files if any
        new_job_id = cu_result.metadata.get("job_id", "")
        if cu_result.downloaded_files and destination:
            retrieved = await loop.run_in_executor(
                None,
                lambda: provider.retrieve_files(
                    new_job_id, destination,
                    cu_result.metadata.get("downloads_dir"),
                ),
            )
            if retrieved:
                files_msg = "\n".join(f"• <code>{html.escape(f)}</code>" for f in retrieved)
                await telegram.send_message(
                    f"📁 <b>Fichiers récupérés</b>\n\n{files_msg}",
                    chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
                    message_thread_id=thread_id,
                )

        # Update pending state for potential further interactions
        state.pending_computer_use[new_job_id] = {
            "session_id": cu_result.session_id,
            "thread_id": thread_id,
            "provider_config": pending["provider_config"],
            "destination": destination,
            "task": pending["task"],
        }
        # Clean old job
        if job_id != new_job_id:
            state.pending_computer_use.pop(job_id, None)

        # Send result with buttons for further interaction
        buttons = {
            "inline_keyboard": [[
                {"text": "✅ Continuer", "callback_data": f"cu:confirm:{new_job_id}"},
                {"text": "✅ Terminé", "callback_data": f"cu:done:{new_job_id}"},
                {"text": "❌ Annuler", "callback_data": f"cu:cancel:{new_job_id}"},
            ]]
        }

        await _send_computer_use_result(
            cu_result, provider, buttons,
            chat_id, bot, thread_id,
            hint="Utilise les boutons ou écris des instructions.",
        )

    elif action == "done":
        # User marks task as complete — cleanup
        state.pending_computer_use.pop(job_id, None)
        await telegram.send_message(
            "✅ Tâche terminée. Topic clos.",
            chat_id=chat_id, parse_mode="HTML", api_url=bot.api_url,
            message_thread_id=thread_id,
        )
