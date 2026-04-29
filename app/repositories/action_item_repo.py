import json
import uuid
from typing import List, Optional

from sqlmodel import Session, select

from app.models.db import ActionItem, MeetingSummary, get_engine


def _session(engine):
    return Session(engine, expire_on_commit=False)


class ActionItemRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def create_many(self, meeting_id: str, items: List[dict]) -> List[ActionItem]:
        with _session(self._engine) as session:
            records = [
                ActionItem(
                    id=str(uuid.uuid4()),
                    meeting_id=meeting_id,
                    task=item["task"],
                    deadline=item.get("deadline"),
                    confidence=item.get("confidence", 0.0),
                    needs_review=item.get("needs_review", False),
                    supporting_quote=item.get("supporting_quote", ""),
                    status="pending",
                )
                for item in items
            ]
            for r in records:
                session.add(r)
            session.commit()
            return records

    def list_by_meeting(self, meeting_id: str) -> List[ActionItem]:
        with _session(self._engine) as session:
            return list(session.exec(
                select(ActionItem).where(ActionItem.meeting_id == meeting_id)
            ).all())

    def update_status(self, item_id: str, status: str) -> None:
        with _session(self._engine) as session:
            item = session.get(ActionItem, item_id)
            if item:
                item.status = status
                session.add(item)
                session.commit()

    def mark_sent(self, item_id: str, sent_via: str) -> None:
        with _session(self._engine) as session:
            item = session.get(ActionItem, item_id)
            if item:
                item.sent_via = sent_via
                session.add(item)
                session.commit()


class SummaryRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def upsert(self, meeting_id: str, summary: dict) -> MeetingSummary:
        with _session(self._engine) as session:
            record = session.get(MeetingSummary, meeting_id)
            if record is None:
                record = MeetingSummary(id=meeting_id, meeting_id=meeting_id)
                session.add(record)
            record.decisions_json = json.dumps(summary.get("decisions", []))
            record.blockers_json = json.dumps(summary.get("blockers", []))
            record.commitments_json = json.dumps(summary.get("commitments", []))
            record.next_steps_json = json.dumps(summary.get("next_steps", []))
            session.commit()
            return record

    def get(self, meeting_id: str) -> Optional[MeetingSummary]:
        with _session(self._engine) as session:
            return session.get(MeetingSummary, meeting_id)
