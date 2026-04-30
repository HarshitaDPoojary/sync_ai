from unittest.mock import MagicMock, patch

from app.integrations.google_calendar import GoogleCalendarClient


def test_get_upcoming_events_skips_events_without_datetime():
    timed_event = {
        "id": "evt_timed",
        "summary": "Timed call",
        "hangoutLink": "https://meet.google.com/timed",
        "start": {"dateTime": "2026-04-30T10:00:00+00:00"},
    }
    all_day_event = {
        "id": "evt_all_day",
        "summary": "All day",
        "hangoutLink": "https://meet.google.com/all-day",
        "start": {"date": "2026-04-30"},
    }
    events_resource = MagicMock()
    events_resource.list.return_value.execute.return_value = {"items": [all_day_event, timed_event]}
    service = MagicMock()
    service.events.return_value = events_resource

    with patch("app.integrations.google_calendar.build", return_value=service):
        client = GoogleCalendarClient("access", "refresh")

    events = client.get_upcoming_events()

    assert events == [{
        "id": "evt_timed",
        "title": "Timed call",
        "start_time": "2026-04-30T10:00:00+00:00",
        "meeting_url": "https://meet.google.com/timed",
    }]
