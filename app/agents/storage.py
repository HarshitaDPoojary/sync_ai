import json
import uuid
from typing import Any, Optional

from sqlmodel import Session

from app.core.config import get_settings
from app.core.graph import MeetingState
from app.models.db import Transcript, get_engine


def run_storage_node(
    state: MeetingState,
    engine=None,
    chroma_collection=None,
    embedder=None,
) -> MeetingState:
    if engine is None:
        engine = get_engine()

    chunks = state["transcript_chunks"]
    full_text = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)

    with Session(engine) as session:
        transcript = session.get(Transcript, state["meeting_id"])
        if transcript is None:
            transcript = Transcript(id=state["meeting_id"], meeting_id=state["meeting_id"])
            session.add(transcript)
        transcript.full_text = full_text
        transcript.chunks_json = json.dumps(chunks)
        session.commit()

    if chroma_collection and embedder:
        words = full_text.split()
        chunk_size, overlap = 200, 50
        step = chunk_size - overlap
        for i, start in enumerate(range(0, max(len(words), 1), step)):
            chunk_text = " ".join(words[start: start + chunk_size])
            if not chunk_text.strip():
                continue
            embedding = embedder.embed_documents([chunk_text])[0]
            chroma_collection.upsert(
                ids=[f"{state['meeting_id']}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{"meeting_id": state["meeting_id"], "sequence": i}],
            )

        for item in state["action_items"]:
            embedding = embedder.embed_documents([item["task"]])[0]
            chroma_collection.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[item["task"]],
                metadatas=[{
                    "meeting_id": state["meeting_id"],
                    "owner_name": item.get("owner_name") or "",
                    "deadline": item.get("deadline") or "",
                }],
            )

    return state
