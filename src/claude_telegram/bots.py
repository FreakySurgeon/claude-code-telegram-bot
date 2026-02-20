"""Bot configuration — routes behavior based on which Telegram bot received the message."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for a single Telegram bot identity."""
    name: str
    token: str
    chat_id: str
    fixed_working_dir: str | None = None
    system_prompt_path: str | None = None
    mcp_config_path: str | None = None
    use_queue: bool = False
    commands_whitelist: list[str] = field(default_factory=list)

    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    @property
    def system_prompt(self) -> str | None:
        if not self.system_prompt_path:
            return None
        try:
            return Path(self.system_prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read system prompt {self.system_prompt_path}: {e}")
            return None

    def is_authorized(self, chat_id: str | int) -> bool:
        return str(chat_id) == str(self.chat_id)


def create_bots() -> dict[str, BotConfig]:
    """Create bot configurations from settings."""
    bots = {}

    # Dev bot (always present)
    bots["dev"] = BotConfig(
        name="dev",
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        use_queue=False,
        commands_whitelist=[
            "/start", "/help", "/c", "/continue", "/new", "/dir", "/dirs",
            "/repos", "/rmdir", "/compact", "/cancel", "/status",
        ],
    )

    # GTD bot (optional)
    if settings.gtd_bot_token and settings.gtd_chat_id:
        bots["gtd"] = BotConfig(
            name="gtd",
            token=settings.gtd_bot_token,
            chat_id=settings.gtd_chat_id,
            fixed_working_dir=settings.gtd_working_dir,
            system_prompt_path=settings.gtd_prompt_path,
            mcp_config_path=settings.gtd_mcp_config,
            use_queue=True,
            commands_whitelist=[
                "/start", "/help", "/new", "/compact", "/cancel", "/status",
            ],
        )

    return bots
