import pytest
from pytest_httpx import HTTPXMock
from app.core.recall import RecallClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RECALL_BASE_URL", "https://us-east-1.recall.ai/api/v1")
    return RecallClient(api_key="test_key")


@pytest.mark.asyncio
async def test_create_bot(httpx_mock: HTTPXMock, client):
    httpx_mock.add_response(
        method="POST",
        url="https://us-east-1.recall.ai/api/v1/bot/",
        json={"id": "bot_123"},
        status_code=201,
    )
    bot = await client.create_bot_with_webhook(
        meeting_url="https://zoom.us/j/999",
        webhook_url="https://app.com/webhook/recall/m1",
    )
    assert bot["id"] == "bot_123"


@pytest.mark.asyncio
async def test_remove_bot(httpx_mock: HTTPXMock, client):
    httpx_mock.add_response(
        method="DELETE",
        url="https://us-east-1.recall.ai/api/v1/bot/bot_123/leave_call/",
        json={},
        status_code=200,
    )
    await client.remove_bot("bot_123")
