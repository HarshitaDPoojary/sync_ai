from unittest.mock import MagicMock
from sqlmodel import Session
from app.agents.storage import run_storage_node
from app.core.graph import make_initial_state, TranscriptChunk
from app.models.db import Transcript


def test_storage_node_writes_transcript(engine):
    state = make_initial_state("m1", [])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Hello there", timestamp=0.0, sequence_num=1)
    ]
    state["action_items"] = []
    state["summary"] = None

    run_storage_node(state, engine=engine)

    with Session(engine) as session:
        transcript = session.get(Transcript, "m1")
    assert transcript is not None
    assert "Hello there" in transcript.full_text


def test_storage_node_embeds_chunks_when_chroma_provided(engine):
    state = make_initial_state("m1", [])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Hello there", timestamp=0.0, sequence_num=1)
    ]
    state["action_items"] = []
    state["summary"] = None

    mock_chroma = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 384]

    run_storage_node(state, engine=engine, chroma_collection=mock_chroma, embedder=mock_embedder)

    mock_chroma.upsert.assert_called()
