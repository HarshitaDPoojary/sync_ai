from app.repositories.meeting_repo import MeetingRepository
from app.repositories.transcript_repo import TranscriptRepository
from app.repositories.action_item_repo import ActionItemRepository, SummaryRepository


def test_meeting_repo_create_and_get(engine):
    repo = MeetingRepository(engine)
    repo.create("m1", "Sprint Review", "https://zoom.us/j/1", "bot_1")
    meeting = repo.get("m1")
    assert meeting.title == "Sprint Review"
    assert meeting.status == "active"


def test_meeting_repo_mark_ended(engine):
    repo = MeetingRepository(engine)
    repo.create("m2", "Standup", "https://zoom.us/j/2", "bot_2")
    repo.mark_ended("m2")
    assert repo.get("m2").status == "ended"


def test_meeting_repo_set_trace_url(engine):
    repo = MeetingRepository(engine)
    repo.create("m3", "Design", "https://zoom.us/j/3", "bot_3")
    repo.set_trace_url("m3", "https://smith.langchain.com/trace/abc")
    assert repo.get_trace_url("m3") == "https://smith.langchain.com/trace/abc"


def test_meeting_repo_add_participants(engine):
    repo = MeetingRepository(engine)
    repo.create("m4", "Planning", "https://zoom.us/j/4", "bot_4")
    participants = repo.add_participants("m4", ["alice@x.com", "bob@x.com"])
    assert len(participants) == 2
    assert participants[0].display_name == "alice"
    assert participants[1].email == "bob@x.com"


def test_transcript_repo_upsert_and_get(engine):
    MeetingRepository(engine).create("m5", "T", "u", "b")
    repo = TranscriptRepository(engine)
    repo.upsert("m5", "Alice: Hello", [{"speaker": "Alice", "text": "Hello", "timestamp": 0.0, "sequence_num": 1}])
    transcript = repo.get("m5")
    assert "Hello" in transcript.full_text


def test_transcript_repo_upsert_is_idempotent(engine):
    MeetingRepository(engine).create("m6", "T", "u", "b")
    repo = TranscriptRepository(engine)
    repo.upsert("m6", "First", [])
    repo.upsert("m6", "Second", [])
    assert repo.get("m6").full_text == "Second"


def test_action_item_repo_create_and_list(engine):
    MeetingRepository(engine).create("m7", "T", "u", "b")
    repo = ActionItemRepository(engine)
    repo.create_many("m7", [
        {"task": "Deploy API", "confidence": 0.9, "needs_review": False, "supporting_quote": "we will deploy"},
        {"task": "Write tests", "confidence": 0.8, "needs_review": True, "supporting_quote": ""},
    ])
    items = repo.list_by_meeting("m7")
    assert len(items) == 2
    tasks = {i.task for i in items}
    assert "Deploy API" in tasks


def test_action_item_repo_update_status(engine):
    MeetingRepository(engine).create("m8", "T", "u", "b")
    repo = ActionItemRepository(engine)
    created = repo.create_many("m8", [{"task": "Fix bug", "confidence": 0.7, "needs_review": False, "supporting_quote": ""}])
    repo.update_status(created[0].id, "accepted")
    items = repo.list_by_meeting("m8")
    assert items[0].status == "accepted"


def test_summary_repo_upsert(engine):
    MeetingRepository(engine).create("m9", "T", "u", "b")
    repo = SummaryRepository(engine)
    repo.upsert("m9", {
        "decisions": ["Use PostgreSQL"],
        "blockers": ["No credentials"],
        "commitments": [],
        "next_steps": ["Deploy by Friday"],
    })
    summary = repo.get("m9")
    assert "Use PostgreSQL" in summary.decisions
    assert "No credentials" in summary.blockers
