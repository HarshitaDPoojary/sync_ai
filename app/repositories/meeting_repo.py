import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Session, select

from app.models.db import Meeting, Participant, get_engine


def _session(engine):
    return Session(engine, expire_on_commit=False)


class MeetingRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def create(
        self,
        meeting_id: str,
        title: str,
        platform_url: str,
        recall_bot_id: str,
        user_id: Optional[str] = None,
        slack_channel_id: Optional[str] = None,
    ) -> Meeting:
        with _session(self._engine) as session:
            meeting = Meeting(
                id=meeting_id,
                title=title,
                platform_url=platform_url,
                status="active",
                recall_bot_id=recall_bot_id,
                slack_channel_id=slack_channel_id,
                user_id=user_id,
            )
            session.add(meeting)
            session.commit()
            return meeting

    def get(self, meeting_id: str, user_id: Optional[str] = None) -> Optional[Meeting]:
        with _session(self._engine) as session:
            meeting = session.get(Meeting, meeting_id)
            if meeting and user_id and meeting.user_id and meeting.user_id != user_id:
                return None  # treat as not found — don't leak existence
            return meeting

    def list_by_user(self, user_id: str) -> List[Meeting]:
        with _session(self._engine) as session:
            return list(session.exec(
                select(Meeting)
                .where(Meeting.user_id == user_id)
                .order_by(Meeting.started_at.desc())
            ).all())

    def mark_ended(self, meeting_id: str) -> None:
        with _session(self._engine) as session:
            meeting = session.get(Meeting, meeting_id)
            if meeting:
                meeting.status = "ended"
                meeting.ended_at = datetime.now(timezone.utc)
                session.add(meeting)
                session.commit()

    def set_trace_url(self, meeting_id: str, trace_url: str) -> None:
        with _session(self._engine) as session:
            meeting = session.get(Meeting, meeting_id)
            if meeting:
                meeting.langsmith_trace_url = trace_url
                session.add(meeting)
                session.commit()

    def get_trace_url(self, meeting_id: str) -> Optional[str]:
        meeting = self.get(meeting_id)
        return meeting.langsmith_trace_url if meeting else None

    def add_participants(self, meeting_id: str, emails: List[str]) -> List[Participant]:
        with _session(self._engine) as session:
            participants = [
                Participant(
                    id=str(uuid.uuid4()),
                    meeting_id=meeting_id,
                    display_name=email.split("@")[0],
                    email=email,
                )
                for email in emails
            ]
            for p in participants:
                session.add(p)
            session.commit()
            return participants
