import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models.db import UserIntegration, get_engine
from app.repositories.calendar_event_repo import CalendarEventRepository
from app.repositories.meeting_repo import MeetingRepository
from app.repositories.user_repo import UserRepository

logger = logging.getLogger("sync_ai.calendar_poller")


async def poll_all_users_calendars(sessions: Dict) -> None:
    while True:
        try:
            await _do_poll(sessions)
        except Exception as e:
            logger.warning("calendar_poll_error error=%s", e)
        await asyncio.sleep(get_settings().calendar_poll_interval_seconds)


async def _do_poll(sessions: Dict) -> None:
    from app.integrations.google_calendar import GoogleCalendarClient
    settings = get_settings()
    engine = get_engine()

    with Session(engine) as db:
        integrations = list(db.exec(
            select(UserIntegration).where(UserIntegration.provider == "google_calendar")
        ).all())

    cal_repo = CalendarEventRepository()
    user_repo = UserRepository()
    offset_seconds = settings.calendar_bot_dispatch_offset_seconds

    for integration in integrations:
        try:
            client = GoogleCalendarClient(integration.access_token, integration.refresh_token)
            events = client.get_upcoming_events(lookahead_minutes=60)
        except Exception as e:
            logger.warning("calendar_fetch_failed user_id=%s error=%s", integration.user_id, e)
            continue

        for event in events:
            start_time = datetime.fromisoformat(event["start_time"])
            cal_repo.upsert(
                event_id=event["id"],
                user_id=integration.user_id,
                title=event["title"],
                start_time=start_time,
                meeting_url=event["meeting_url"],
            )

        dispatch_cutoff = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
        undispatched = cal_repo.get_undispatched(integration.user_id, dispatch_cutoff)

        for cal_event in undispatched:
            user = user_repo.get_by_id(integration.user_id)
            if not user:
                continue
            meeting_id = str(uuid.uuid4())
            try:
                from app.core.session import MeetingSession
                slack_int = user_repo.get_integration(user.id, "slack")
                slack_token = slack_int.access_token if slack_int else None
                session = MeetingSession(
                    meeting_id=meeting_id,
                    meeting_url=cal_event.meeting_url,
                    title=cal_event.title,
                    participant_emails=[user.email],
                    slack_bot_token=slack_token,
                    user_id=user.id,
                )
                await session.start()
                MeetingRepository().create(
                    meeting_id=meeting_id,
                    title=cal_event.title,
                    platform_url=cal_event.meeting_url,
                    recall_bot_id=session.bot_id or "",
                    user_id=user.id,
                )
                sessions[meeting_id] = session
                cal_repo.mark_dispatched(cal_event.id, meeting_id)
                logger.info("calendar_bot_dispatched user_id=%s meeting_id=%s event=%s",
                            user.id, meeting_id, cal_event.id)
            except Exception as e:
                logger.warning("calendar_dispatch_failed user_id=%s event_id=%s error=%s",
                               integration.user_id, cal_event.id, e)
