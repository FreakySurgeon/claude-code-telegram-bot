"""Telegram bot service."""

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from .config import settings

logger = logging.getLogger(__name__)

DEFAULT_API_URL = f"https://api.telegram.org/bot{settings.telegram_bot_token}"


async def send_message(
    text: str,
    chat_id: str | None = None,
    parse_mode: str = "Markdown",
    reply_markup: dict | None = None,
    api_url: str | None = None,
    message_thread_id: int | None = None,
) -> dict:
    """Send a message to Telegram."""
    api = api_url or DEFAULT_API_URL
    chat_id = chat_id or settings.telegram_chat_id

    # Telegram requires non-empty text
    if not text or not text.strip():
        text = "(empty)"

    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    # Only add parse_mode if specified (can cause issues with special chars)
    if parse_mode:
        payload["parse_mode"] = parse_mode

    if reply_markup:
        payload["reply_markup"] = reply_markup

    if message_thread_id is not None:
        payload["message_thread_id"] = message_thread_id

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/sendMessage", json=payload)
        if response.status_code != 200:
            logger.error(f"Telegram error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()


async def edit_message(
    message_id: int,
    text: str,
    chat_id: str | None = None,
    parse_mode: str = "Markdown",
    api_url: str | None = None,
    message_thread_id: int | None = None,
    reply_markup: dict | None = None,
) -> dict:
    """Edit an existing message."""
    api = api_url or DEFAULT_API_URL
    chat_id = chat_id or settings.telegram_chat_id

    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    if message_thread_id is not None:
        payload["message_thread_id"] = message_thread_id

    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/editMessageText", json=payload)
        response.raise_for_status()
        return response.json()


async def delete_message(chat_id: str | int, message_id: int, api_url: str | None = None) -> dict:
    """Delete a message."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
        )
        response.raise_for_status()
        return response.json()


async def set_webhook(url: str, api_url: str | None = None) -> dict:
    """Set the Telegram webhook URL."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api}/setWebhook",
            json={"url": url, "allowed_updates": ["message", "callback_query"]},
        )
        response.raise_for_status()
        return response.json()


@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def set_webhook_with_retry(url: str, api_url: str | None = None) -> dict:
    """Set webhook with exponential backoff retry for DNS propagation."""
    return await set_webhook(url, api_url=api_url)


async def delete_webhook(api_url: str | None = None) -> dict:
    """Delete the Telegram webhook."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/deleteWebhook")
        response.raise_for_status()
        return response.json()


async def get_updates(offset: int = 0, timeout: int = 30, api_url: str | None = None) -> list[dict]:
    """Get updates using long polling."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient(timeout=timeout + 10) as client:
        response = await client.post(
            f"{api}/getUpdates",
            json={
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])


async def answer_callback(callback_query_id: str, text: str | None = None, api_url: str | None = None) -> dict:
    """Answer a callback query (inline button press)."""
    api = api_url or DEFAULT_API_URL
    payload = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/answerCallbackQuery", json=payload)
        response.raise_for_status()
        return response.json()


async def get_file(file_id: str, api_url: str | None = None) -> dict:
    """Get file info for downloading."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/getFile", json={"file_id": file_id})
        response.raise_for_status()
        return response.json()


async def download_file(file_path: str, api_url: str | None = None) -> bytes:
    """Download a file from Telegram servers."""
    api = api_url or DEFAULT_API_URL
    token = api.split("/bot")[1]
    url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def create_forum_topic(
    chat_id: str | int,
    name: str,
    api_url: str | None = None,
) -> dict:
    """Create a forum topic in a supergroup chat."""
    api = api_url or DEFAULT_API_URL
    payload = {
        "chat_id": chat_id,
        "name": name[:128],
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/createForumTopic", json=payload)
        if response.status_code != 200:
            logger.error(f"createForumTopic error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()


async def edit_forum_topic(
    chat_id: str | int,
    message_thread_id: int,
    name: str,
    api_url: str | None = None,
) -> dict:
    """Edit a forum topic name in a supergroup chat."""
    api = api_url or DEFAULT_API_URL
    payload = {
        "chat_id": chat_id,
        "message_thread_id": message_thread_id,
        "name": name[:128],
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/editForumTopic", json=payload)
        if response.status_code != 200:
            logger.error(f"editForumTopic error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()


async def get_chat(
    chat_id: str | int,
    api_url: str | None = None,
) -> dict:
    """Get chat information."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/getChat", json={"chat_id": chat_id})
        if response.status_code != 200:
            logger.error(f"getChat error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()


async def get_me(api_url: str | None = None) -> dict:
    """Get bot info via getMe."""
    api = api_url or DEFAULT_API_URL
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api}/getMe")
        response.raise_for_status()
        return response.json()


def is_authorized(chat_id: str | int) -> bool:
    """Check if the chat is authorized."""
    return str(chat_id) == str(settings.telegram_chat_id)
