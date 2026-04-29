import hashlib
from typing import Any, Optional

from app.core.graph import MeetingState
from app.repositories.transcript_repo import TranscriptRepository


def run_storage_node(
    state: MeetingState,
    engine=None,
    chroma_collection=None,
    embedder=None,
) -> MeetingState:
    chunks = state["transcript_chunks"]
    full_text = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)

    TranscriptRepository(engine).upsert(
        meeting_id=state["meeting_id"],
        full_text=full_text,
        chunks=list(chunks),
    )

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
            item_id = hashlib.sha256(
                f"{state['meeting_id']}:{item['task']}".encode()
            ).hexdigest()[:32]
            chroma_collection.upsert(
                ids=[item_id],
                embeddings=[embedding],
                documents=[item["task"]],
                metadatas=[{
                    "meeting_id": state["meeting_id"],
                    "owner_name": item.get("owner_name") or "",
                    "deadline": item.get("deadline") or "",
                }],
            )

    return state
