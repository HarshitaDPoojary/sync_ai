import pytest

from app.core.config import ConfigurationError, Settings, validate_settings


def _settings(**overrides) -> Settings:
    isolated_defaults = {
        "recall_api_key": "",
        "groq_api_key": "",
        "huggingface_api_key": "",
        "clerk_secret_key": "",
        "clerk_publishable_key": "",
        "clerk_frontend_api": "",
        "slack_client_id": "",
        "slack_client_secret": "",
        "slack_oauth_redirect_uri": "",
        "slack_bot_token": "",
        "slack_channel_id": "",
        "google_client_id": "",
        "google_client_secret": "",
        "google_oauth_redirect_uri": "",
        "database_url": "",
        "chroma_persist_dir": "",
        "webhook_base_url": "",
        "recall_webhook_secret": "",
    }
    return Settings(_env_file=None, **{**isolated_defaults, **overrides})


def test_development_allows_missing_external_provider_settings():
    validate_settings(_settings(app_env="development"))


def test_production_requires_core_settings():
    with pytest.raises(ConfigurationError) as exc_info:
        validate_settings(_settings(app_env="production"))

    message = str(exc_info.value)
    assert "Missing production settings" in message
    assert "clerk_secret_key" in message
    assert "recall_webhook_secret" in message


def test_production_accepts_complete_core_settings():
    validate_settings(_settings(
        app_env="production",
        recall_api_key="recall",
        groq_api_key="groq",
        huggingface_api_key="hf",
        clerk_secret_key="clerk_secret",
        clerk_publishable_key="clerk_public",
        clerk_frontend_api="clerk.example.com",
        database_url="./data/sync_ai.db",
        chroma_persist_dir="./data/chroma",
        webhook_base_url="https://app.example.com",
        recall_webhook_secret="whsec_d2ViaG9va19zZWNyZXRfdGVzdA==",
    ))


def test_production_rejects_malformed_recall_webhook_secret():
    with pytest.raises(ConfigurationError) as exc_info:
        validate_settings(_settings(
            app_env="production",
            recall_api_key="recall",
            groq_api_key="groq",
            huggingface_api_key="hf",
            clerk_secret_key="clerk_secret",
            clerk_publishable_key="clerk_public",
            clerk_frontend_api="clerk.example.com",
            database_url="./data/sync_ai.db",
            chroma_persist_dir="./data/chroma",
            webhook_base_url="https://app.example.com",
            recall_webhook_secret="not-a-recall-secret",
        ))

    assert "starting with whsec_" in str(exc_info.value)


def test_partial_optional_integrations_are_rejected():
    with pytest.raises(ConfigurationError) as exc_info:
        validate_settings(_settings(slack_client_id="client_id"))

    message = str(exc_info.value)
    assert "Incomplete Slack OAuth settings" in message
    assert "slack_client_secret" in message
    assert "slack_oauth_redirect_uri" in message
