from functools import lru_cache
from typing import Iterable

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Recall.ai
    recall_api_key: str = ""
    recall_base_url: str = "https://us-east-1.recall.ai/api/v1"
    recall_transcription_provider: str = "assembly_ai"
    recall_bot_name: str = "Meeting Copilot"

    # Groq (swap groq_model to switch LLM, e.g. "llama-3.3-70b-versatile")
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"
    groq_analysis_temperature: float = 0.3
    groq_extraction_temperature: float = 0.1

    # LangSmith
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "sync_ai"
    langsmith_signal_dataset: str = "sync_ai-signal-detection"
    langsmith_extraction_dataset: str = "sync_ai-action-extraction"

    # Clerk auth
    clerk_secret_key: str = ""
    clerk_publishable_key: str = ""
    clerk_frontend_api: str = ""   # e.g. "clerk.your-domain.com"

    # Slack (legacy single-tenant fallback — replaced by per-user OAuth tokens)
    slack_bot_token: str = ""
    slack_channel_id: str = ""

    # Slack OAuth app (for per-user "Add to Slack" flow)
    slack_client_id: str = ""
    slack_client_secret: str = ""
    slack_oauth_redirect_uri: str = ""

    # Google OAuth app (Gmail send + Calendar read — one app, two scopes)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_oauth_redirect_uri: str = ""

    # Gmail (legacy single-tenant fallback — replaced by per-user Google OAuth)
    gmail_credentials_json: str = ""

    # Google Calendar auto-join
    calendar_poll_interval_seconds: int = 300    # poll every 5 minutes
    calendar_bot_dispatch_offset_seconds: int = 120  # join 2 min before start

    # Database
    database_url: str = "./data/sync_ai.db"
    chroma_persist_dir: str = "./data/chroma"

    # App
    app_env: str = "development"
    webhook_base_url: str = "http://localhost:8000"
    recall_webhook_secret: str = ""  # shared secret for Recall.ai webhook HMAC verification
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    # Embeddings — uses HuggingFace Inference API (no local model, no RAM cost)
    huggingface_api_key: str = ""
    embedding_model: str = "thenlper/gte-large"

    # Eval datasets (HuggingFace repo names)
    ami_dataset_repo: str = "edinburghcristin/ami-corpus"
    meetingbank_dataset_repo: str = "huuuyeah/meetingbank"
    eval_fixtures_path: str = "tests/eval/datasets/golden_transcripts.json"

    # Analysis tuning
    analysis_token_threshold: int = 500
    analysis_window_seconds: int = 300
    extraction_checkpoint_chunks: int = 50

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}


def _missing(settings: Settings, names: Iterable[str]) -> list[str]:
    return [name for name in names if not str(getattr(settings, name, "")).strip()]


def _configured(settings: Settings, names: Iterable[str]) -> list[str]:
    return [name for name in names if str(getattr(settings, name, "")).strip()]


def validate_settings(settings: Settings) -> None:
    """Validate environment-driven configuration.

    Development stays permissive so tests and local UI work without every external
    provider configured. Production fails fast for core dependencies, and all
    environments reject partially configured optional integrations.
    """
    errors: list[str] = []

    if settings.is_production:
        required = [
            "recall_api_key",
            "groq_api_key",
            "huggingface_api_key",
            "clerk_secret_key",
            "clerk_publishable_key",
            "clerk_frontend_api",
            "database_url",
            "chroma_persist_dir",
            "webhook_base_url",
            "recall_webhook_secret",
        ]
        missing = _missing(settings, required)
        if missing:
            errors.append("Missing production settings: " + ", ".join(sorted(missing)))
        elif not settings.recall_webhook_secret.startswith("whsec_"):
            errors.append("recall_webhook_secret must be a Recall workspace secret starting with whsec_")

    optional_groups = {
        "Slack OAuth": [
            "slack_client_id",
            "slack_client_secret",
            "slack_oauth_redirect_uri",
        ],
        "Slack legacy bot": [
            "slack_bot_token",
            "slack_channel_id",
        ],
        "Google OAuth": [
            "google_client_id",
            "google_client_secret",
            "google_oauth_redirect_uri",
        ],
    }
    for label, names in optional_groups.items():
        if _configured(settings, names):
            missing = _missing(settings, names)
            if missing:
                errors.append(f"Incomplete {label} settings: " + ", ".join(sorted(missing)))

    if errors:
        raise ConfigurationError("; ".join(errors))


@lru_cache
def get_settings() -> Settings:
    return Settings()
