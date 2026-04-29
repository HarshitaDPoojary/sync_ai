from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Recall.ai
    recall_api_key: str = ""
    recall_base_url: str = "https://us-east-1.recall.ai/api/v1"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # LangSmith
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "sync_ai"

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
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Analysis tuning
    analysis_token_threshold: int = 500
    analysis_window_seconds: int = 300
    extraction_checkpoint_chunks: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
