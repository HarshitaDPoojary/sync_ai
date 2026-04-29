import json
from typing import List, Optional

from sqlmodel import Session

from app.models.db import Transcript, get_engine


def _session(engine):
    return Session(engine, expire_on_commit=False)


class TranscriptRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def upsert(self, meeting_id: str, full_text: str, chunks: List[dict]) -> Transcript:
        with _session(self._engine) as session:
            transcript = session.get(Transcript, meeting_id)
            if transcript is None:
                transcript = Transcript(id=meeting_id, meeting_id=meeting_id)
                session.add(transcript)
            transcript.full_text = full_text
            transcript.chunks_json = json.dumps(chunks)
            session.commit()
            return transcript

    def get(self, meeting_id: str) -> Optional[Transcript]:
        with _session(self._engine) as session:
            return session.get(Transcript, meeting_id)
