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

    # Slack
    slack_bot_token: str = ""
    slack_channel_id: str = ""

    # Gmail
    gmail_credentials_json: str = ""

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
