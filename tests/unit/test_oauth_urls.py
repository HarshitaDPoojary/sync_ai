from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.auth.clerk import get_current_user
from app.core.config import get_settings
from app.main import app
from app.models.db import User


def _auth_user():
    return User(id="user_1", clerk_user_id="clerk_1", email="user@example.com", name="User")


def test_google_oauth_start_url_encodes_query_params(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "test_secret")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "google-secret")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://example.com/auth/google/callback")
    get_settings.cache_clear()
    app.dependency_overrides[get_current_user] = _auth_user

    try:
        with TestClient(app, follow_redirects=False) as client:
            response = client.get("/auth/google")
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 307
    location = response.headers["location"]
    assert " " not in location

    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    assert parsed.netloc == "accounts.google.com"
    assert params["client_id"] == ["google-client"]
    assert params["redirect_uri"] == ["https://example.com/auth/google/callback"]
    assert params["scope"] == [
        "https://www.googleapis.com/auth/gmail.send "
        "https://www.googleapis.com/auth/calendar.readonly"
    ]


def test_slack_oauth_start_url_encodes_query_params(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "test_secret")
    monkeypatch.setenv("SLACK_CLIENT_ID", "slack-client")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "slack-secret")
    monkeypatch.setenv("SLACK_OAUTH_REDIRECT_URI", "https://example.com/auth/slack/callback")
    get_settings.cache_clear()
    app.dependency_overrides[get_current_user] = _auth_user

    try:
        with TestClient(app, follow_redirects=False) as client:
            response = client.get("/auth/slack")
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 307
    location = response.headers["location"]
    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    assert parsed.netloc == "slack.com"
    assert params["client_id"] == ["slack-client"]
    assert params["redirect_uri"] == ["https://example.com/auth/slack/callback"]
    assert params["scope"] == ["chat:write,chat:write.public,channels:read"]
