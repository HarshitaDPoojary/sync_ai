from datetime import datetime
from typing import List, Optional

from sqlmodel import Session, select

from app.models.db import CalendarEvent, get_engine


def _session(engine):
    return Session(engine, expire_on_commit=False)


class CalendarEventRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def upsert(
        self,
        event_id: str,
        user_id: str,
        title: str,
        start_time: datetime,
        meeting_url: str,
    ) -> CalendarEvent:
        with _session(self._engine) as session:
            existing = session.get(CalendarEvent, event_id)
            if existing:
                existing.title = title
                existing.start_time = start_time
                existing.meeting_url = meeting_url
                session.add(existing)
                session.commit()
                return existing
            event = CalendarEvent(
                id=event_id,
                user_id=user_id,
                title=title,
                start_time=start_time,
                meeting_url=meeting_url,
            )
            session.add(event)
            session.commit()
            return event

    def get_undispatched(self, user_id: str, cutoff: datetime) -> List[CalendarEvent]:
        with _session(self._engine) as session:
            return list(session.exec(
                select(CalendarEvent)
                .where(CalendarEvent.user_id == user_id)
                .where(CalendarEvent.bot_dispatched == False)
                .where(CalendarEvent.start_time <= cutoff)
            ).all())

    def mark_dispatched(self, event_id: str, meeting_id: str) -> None:
        with _session(self._engine) as session:
            event = session.get(CalendarEvent, event_id)
            if event:
                event.bot_dispatched = True
                event.meeting_id = meeting_id
                session.add(event)
                session.commit()
