from unittest.mock import MagicMock, patch
from app.agents.analysis import run_analysis_node
from app.core.graph import make_initial_state, TranscriptChunk

FAKE_LLM_OUTPUT = {
    "suggestions": [{"type": "question", "text": "What is the deadline?", "confidence": 0.9}],
    "signals": [{"signal_type": "decision", "summary": "Use PostgreSQL", "speaker": "Alice", "timestamp": 10.0}],
}


def test_analysis_node_updates_state():
    state = make_initial_state("m1", [])
    state["analysis_window"] = [
        TranscriptChunk(speaker="Alice", text="We decided to use PostgreSQL", timestamp=10.0, sequence_num=1)
    ]
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = FAKE_LLM_OUTPUT

    with patch("app.agents.analysis.get_analysis_chain", return_value=mock_chain):
        new_state = run_analysis_node(state)

    assert len(new_state["suggestions"]) == 1
    assert new_state["suggestions"][0]["text"] == "What is the deadline?"
    assert len(new_state["signals"]) == 1
    assert new_state["signals"][0]["signal_type"] == "decision"
    assert new_state["last_analysis_token_count"] == 600


def test_analysis_node_skips_empty_window():
    state = make_initial_state("m1", [])
    result = run_analysis_node(state)
    assert result["suggestions"] == []
