from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import get_settings


class TranscriptChunk(TypedDict):
    speaker: str
    text: str
    timestamp: float
    sequence_num: int


class Suggestion(TypedDict):
    suggestion_type: str  # "question" | "clarification" | "talking_point"
    text: str
    confidence: float


class Signal(TypedDict):
    signal_type: str  # "decision" | "blocker" | "commitment"
    summary: str
    speaker: str
    timestamp: float


class ExtractedActionItem(TypedDict):
    task: str
    owner_name: Optional[str]
    owner_email: Optional[str]
    deadline: Optional[str]
    confidence: float
    supporting_quote: str
    needs_review: bool


class MeetingState(TypedDict):
    meeting_id: str
    participants: List[Dict[str, Any]]
    transcript_chunks: List[TranscriptChunk]
    analysis_window: List[TranscriptChunk]  # last N seconds of transcript
    token_count: int
    last_analysis_token_count: int
    suggestions: List[Suggestion]
    signals: List[Signal]
    action_items: List[ExtractedActionItem]
    summary: Optional[Dict[str, List[str]]]
    meeting_ended: bool
    last_checkpoint_chunk_count: int


def make_initial_state(meeting_id: str, participants: List[Dict[str, Any]]) -> MeetingState:
    return MeetingState(
        meeting_id=meeting_id,
        participants=participants,
        transcript_chunks=[],
        analysis_window=[],
        token_count=0,
        last_analysis_token_count=0,
        suggestions=[],
        signals=[],
        action_items=[],
        summary=None,
        meeting_ended=False,
        last_checkpoint_chunk_count=0,
    )


def should_analyze(state: MeetingState) -> bool:
    settings = get_settings()
    return (state["token_count"] - state["last_analysis_token_count"]) >= settings.analysis_token_threshold


def should_extract(state: MeetingState) -> bool:
    settings = get_settings()
    chunks_since_checkpoint = len(state["transcript_chunks"]) - state["last_checkpoint_chunk_count"]
    return state["meeting_ended"] or chunks_since_checkpoint >= settings.extraction_checkpoint_chunks


def build_graph(analysis_node, extraction_node, delivery_node, storage_node) -> StateGraph:
    """
    Builds the LangGraph StateGraph. Each node: (state: MeetingState) -> MeetingState
    """
    graph = StateGraph(MeetingState)

    graph.add_node("analyze", analysis_node)
    graph.add_node("extract", extraction_node)
    graph.add_node("deliver", delivery_node)
    graph.add_node("store", storage_node)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze",
        lambda s: "extract" if should_extract(s) else END,
    )
    graph.add_conditional_edges(
        "extract",
        lambda s: "deliver" if s["meeting_ended"] else END,
    )
    graph.add_edge("deliver", "store")
    graph.add_edge("store", END)

    return graph.compile()
