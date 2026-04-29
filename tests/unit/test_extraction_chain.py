from unittest.mock import MagicMock, patch
from app.agents.extraction import run_extraction_node
from app.core.graph import make_initial_state, TranscriptChunk


def test_extraction_node_verifies_quote():
    state = make_initial_state("m1", [{"name": "Bob", "email": "bob@x.com"}])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Bob will deploy the API by Friday.", timestamp=0.0, sequence_num=1),
        TranscriptChunk(speaker="Bob", text="Agreed.", timestamp=5.0, sequence_num=2),
    ]
    state["meeting_ended"] = True

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "action_items": [{
            "task": "Deploy the API",
            "owner_name": "Bob",
            "deadline": "Friday",
            "confidence": 0.95,
            "supporting_quote": "Bob will deploy the API by Friday",
        }],
        "summary": {"decisions": ["Deploy API by Friday"], "blockers": [], "commitments": [], "next_steps": []},
    }

    with patch("app.agents.extraction.get_extraction_chain", return_value=mock_chain):
        new_state = run_extraction_node(state)

    assert len(new_state["action_items"]) == 1
    item = new_state["action_items"][0]
    assert item["task"] == "Deploy the API"
    assert item["owner_email"] == "bob@x.com"
    assert item["needs_review"] is False


def test_extraction_node_flags_hallucinated_quote():
    state = make_initial_state("m1", [{"name": "Alice", "email": "alice@x.com"}])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Bob will deploy the API by Friday.", timestamp=0.0, sequence_num=1),
    ]
    state["meeting_ended"] = True

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "action_items": [{
            "task": "Redesign database",
            "owner_name": "Alice",
            "deadline": "Monday",
            "confidence": 0.6,
            "supporting_quote": "this quote does not exist in the transcript anywhere",
        }],
        "summary": {"decisions": [], "blockers": [], "commitments": [], "next_steps": []},
    }

    with patch("app.agents.extraction.get_extraction_chain", return_value=mock_chain):
        new_state = run_extraction_node(state)

    assert new_state["action_items"][0]["needs_review"] is True
