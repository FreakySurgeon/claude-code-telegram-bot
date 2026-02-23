"""Configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str  # Allowed chat ID for security

    # Claude
    claude_cli_path: str = "claude"
    claude_working_dir: str | None = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    webhook_path: str = "/webhook"
    webhook_url: str | None = None  # Manual webhook URL (for "webhook" mode)

    # Mode: "polling" (default), "tunnel", or "webhook"
    # - polling: No public URL needed, polls Telegram API (recommended)
    # - tunnel: Auto-creates Cloudflare tunnel
    # - webhook: Use manual webhook_url
    mode: str = "polling"

    # Favorite repos (comma-separated paths relative to home)
    favorite_repos: str = ""

    # GTD Bot (second Telegram bot identity)
    gtd_bot_token: str | None = None
    gtd_chat_id: str | None = None
    gtd_working_dir: str | None = None
    gtd_prompt_path: str | None = None
    gtd_mcp_config: str | None = None
    gtd_cron_prompts_dir: str | None = None
    gtd_post_session_prompt: str | None = None

    # Transcription
    mistral_api_key: str | None = None
    whisper_bin: str = "/opt/whisper.cpp/build/bin/whisper-cli"
    whisper_model: str = "/opt/whisper.cpp/models/ggml-medium.bin"

    # Email webhook
    webhook_secret: str | None = None

    # Trello API (for direct card creation — alerts, enrichment)
    trello_api_key: str = ""
    trello_token: str = ""
    trello_todo_list_id: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_favorite_repos(self) -> list[str]:
        """Parse favorite repos from comma-separated string."""
        if not self.favorite_repos:
            return []
        return [r.strip() for r in self.favorite_repos.split(",") if r.strip()]


settings = Settings()
