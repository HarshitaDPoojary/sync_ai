from sqlmodel import Session
from app.models.db import Meeting, ActionItem


def test_meeting_create(engine):
    with Session(engine) as session:
        meeting = Meeting(
            id="m1", title="Sprint Review",
            platform_url="https://zoom.us/j/123",
            status="active", recall_bot_id="bot_abc",
        )
        session.add(meeting)
        session.commit()
        session.refresh(meeting)
    assert meeting.id == "m1"
    assert meeting.status == "active"


def test_action_item_defaults(engine):
    with Session(engine) as session:
        meeting = Meeting(id="m2", title="t", platform_url="u", status="active", recall_bot_id="b")
        session.add(meeting)
        session.commit()
        item = ActionItem(
            id="a1", meeting_id="m2", task="Deploy API",
            confidence=0.9, status="pending",
            needs_review=False, supporting_quote="We will deploy",
        )
        session.add(item)
        session.commit()
        session.refresh(item)
    assert item.needs_review is False
    assert item.sent_via is None
