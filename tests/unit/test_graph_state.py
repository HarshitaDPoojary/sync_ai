from app.core.graph import MeetingState, make_initial_state, should_analyze


def test_initial_state_structure():
    state = make_initial_state(meeting_id="m1", participants=[])
    assert state["meeting_id"] == "m1"
    assert state["transcript_chunks"] == []
    assert state["suggestions"] == []
    assert state["signals"] == []
    assert state["action_items"] == []
    assert state["summary"] is None
    assert state["meeting_ended"] is False


def test_should_analyze_false_below_threshold():
    state = make_initial_state("m1", [])
    state["token_count"] = 100
    state["last_analysis_token_count"] = 0
    assert should_analyze(state) is False


def test_should_analyze_true_above_threshold():
    state = make_initial_state("m1", [])
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0
    assert should_analyze(state) is True
