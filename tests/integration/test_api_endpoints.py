import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.core.session.RecallClient") as MockRecall, \
         patch("app.core.session.chromadb"), \
         patch("app.core.session.HuggingFaceEmbeddings"), \
         patch("app.core.session.get_engine"), \
         patch("app.repositories.meeting_repo.get_engine"), \
         patch("app.repositories.transcript_repo.get_engine"), \
         patch("app.repositories.action_item_repo.get_engine"), \
         patch("app.main.create_db_and_tables"):
        mock_recall = MagicMock()
        mock_recall.create_bot_with_webhook = AsyncMock(return_value={"id": "bot_test"})
        mock_recall.remove_bot = AsyncMock()
        MockRecall.return_value = mock_recall

        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_webhook_unknown_session(client):
    res = client.post("/webhook/recall/nonexistent", json={
        "transcript": {"words": [{"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker": 0}]}
    })
    assert res.status_code == 200
    assert res.json()["ok"] is False
