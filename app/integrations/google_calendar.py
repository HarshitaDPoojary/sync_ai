from datetime import datetime, timedelta, timezone
from typing import List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.core.config import get_settings


class GoogleCalendarClient:
    def __init__(self, access_token: str, refresh_token: str):
        settings = get_settings()
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
        )
        self._service = build("calendar", "v3", credentials=creds)

    def get_upcoming_events(self, lookahead_minutes: int = 60) -> List[dict]:
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(minutes=lookahead_minutes)
        events = self._service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        result = []
        for e in events.get("items", []):
            url = e.get("hangoutLink") or _extract_video_url(e)
            start_time = e["start"].get("dateTime")
            if url and start_time:
                result.append({
                    "id": e["id"],
                    "title": e.get("summary", "Meeting"),
                    "start_time": start_time,
                    "meeting_url": url,
                })
        return result


def _extract_video_url(event: dict) -> Optional[str]:
    for ep in event.get("conferenceData", {}).get("entryPoints", []):
        if ep.get("entryPointType") == "video":
            return ep.get("uri")
    return None
