import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Optional

from sqlmodel import Field, SQLModel, create_engine

from app.core.config import get_settings


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    clerk_user_id: str = Field(unique=True, index=True)
    email: str = Field(index=True)
    name: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserIntegration(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    provider: str           # "slack" | "gmail" | "google_calendar"
    access_token: str = ""
    refresh_token: str = ""
    token_expires_at: Optional[datetime] = None
    team_id: Optional[str] = None       # Slack workspace ID
    channel_id: Optional[str] = None    # Slack default channel
    extra_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Meeting(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    platform_url: str
    status: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    recall_bot_id: str
    langsmith_trace_url: Optional[str] = None
    slack_channel_id: Optional[str] = None
    user_id: Optional[str] = Field(default=None, foreign_key="user.id", index=True)


class Participant(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    display_name: str
    email: Optional[str] = None
    slack_id: Optional[str] = None


class Transcript(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    full_text: str = ""
    chunks_json: str = "[]"


class ActionItem(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    task: str
    owner_id: Optional[str] = Field(default=None, foreign_key="participant.id")
    deadline: Optional[str] = None
    confidence: float = 0.0
    status: str = "pending"
    needs_review: bool = False
    supporting_quote: str = ""
    sent_via: Optional[str] = None


class MeetingSummary(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    decisions_json: str = "[]"
    blockers_json: str = "[]"
    commitments_json: str = "[]"
    next_steps_json: str = "[]"

    @property
    def decisions(self) -> List[str]:
        return json.loads(self.decisions_json)

    @property
    def blockers(self) -> List[str]:
        return json.loads(self.blockers_json)

    @property
    def commitments(self) -> List[str]:
        return json.loads(self.commitments_json)

    @property
    def next_steps(self) -> List[str]:
        return json.loads(self.next_steps_json)


class CalendarEvent(SQLModel, table=True):
    id: str = Field(primary_key=True)   # Google calendar event id
    user_id: str = Field(foreign_key="user.id", index=True)
    title: str
    meeting_url: Optional[str] = None
    start_time: datetime
    bot_dispatched: bool = False
    meeting_id: Optional[str] = Field(default=None, foreign_key="meeting.id")


def create_db_and_tables(engine=None):
    if engine is None:
        engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def get_engine():
    settings = get_settings()
    return create_engine(
        f"sqlite:///{settings.database_url}",
        connect_args={"check_same_thread": False},
    )
