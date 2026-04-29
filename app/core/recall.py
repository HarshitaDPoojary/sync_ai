from typing import Any, Dict, List

import httpx

from app.core.config import get_settings


class RecallClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        key = api_key or settings.recall_api_key
        self._base_url = settings.recall_base_url
        self._transcription_provider = settings.recall_transcription_provider
        self._headers = {
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        }

    async def create_bot_with_webhook(
        self,
        meeting_url: str,
        webhook_url: str,
        bot_name: str | None = None,
    ) -> Dict[str, Any]:
        if bot_name is None:
            bot_name = get_settings().recall_bot_name
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"{self._base_url}/bot/",
                headers=self._headers,
                json={
                    "meeting_url": meeting_url,
                    "bot_name": bot_name,
                    "transcription_options": {"provider": self._transcription_provider},
                    "real_time_transcription": {
                        "destination_url": webhook_url,
                        "partial_results": False,
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def remove_bot(self, bot_id: str) -> None:
        async with httpx.AsyncClient() as http:
            response = await http.delete(
                f"{self._base_url}/bot/{bot_id}/leave_call/",
                headers=self._headers,
                timeout=30.0,
            )
            response.raise_for_status()

    async def get_participants(self, bot_id: str) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as http:
            response = await http.get(
                f"{self._base_url}/bot/{bot_id}/",
                headers=self._headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("meeting_participants", [])
