"""Configuration settings."""

from pathlib import Path

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

    # Metrics
    cron_token_alert_threshold: int = 100_000  # Alert if cron uses more tokens

    # LLM fallback chain (Claude -> DeepSeek via the Anthropic-compatible endpoint)
    llm_interactive_chain: str = "claude,deepseek"
    llm_provider_state_path: str | None = None  # Shared cooldown file (default: data/llm-provider-state.json)
    deepseek_api_key_file: str | None = None  # Key is read at runtime, never stored in .env
    deepseek_base_url: str = "https://api.deepseek.com/anthropic"
    deepseek_model: str = "deepseek-v4-flash"
    claude_slim_config_dir: str | None = None  # CLAUDE_CONFIG_DIR for DeepSeek runs

    # Channels (ports & adapters): routing file + dotenv used to expand ${VAR} in it
    channel_routing_path: str | None = None
    channel_env_file: str | None = None
    data_dir: str | None = None  # Runtime state (default: $GTD_WORKING_DIR/data)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir).expanduser()
        if self.gtd_working_dir:
            return Path(self.gtd_working_dir).expanduser() / "data"
        return Path("data")

    def get_favorite_repos(self) -> list[str]:
        """Parse favorite repos from comma-separated string."""
        if not self.favorite_repos:
            return []
        return [r.strip() for r in self.favorite_repos.split(",") if r.strip()]


settings = Settings()
