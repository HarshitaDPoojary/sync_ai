from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()
