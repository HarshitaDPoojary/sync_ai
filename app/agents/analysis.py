from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.graph import MeetingState, Signal, Suggestion

_chain_cache = None
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def get_analysis_chain():
    global _chain_cache
    if _chain_cache is None:
        settings = get_settings()
        prompt_text = (_PROMPTS_DIR / "analysis_system.txt").read_text()
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{transcript_window}"),
        ])
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=settings.groq_analysis_temperature,
            api_key=settings.groq_api_key,
        )
        _chain_cache = prompt | llm | JsonOutputParser()
    return _chain_cache


def run_analysis_node(state: MeetingState) -> MeetingState:
    window = state["analysis_window"]
    if not window:
        return state

    window_text = "\n".join(f"{c['speaker']}: {c['text']}" for c in window)
    chain = get_analysis_chain()
    result = chain.invoke({"transcript_window": window_text})

    new_suggestions = [
        Suggestion(
            suggestion_type=s["type"],
            text=s["text"],
            confidence=s.get("confidence", 0.8),
        )
        for s in result.get("suggestions", [])
    ]
    new_signals = [
        Signal(
            signal_type=sig["signal_type"],
            summary=sig["summary"],
            speaker=sig.get("speaker", "Unknown"),
            timestamp=sig.get("timestamp", 0.0),
        )
        for sig in result.get("signals", [])
    ]

    return {
        **state,
        "suggestions": state["suggestions"] + new_suggestions,
        "signals": state["signals"] + new_signals,
        "last_analysis_token_count": state["token_count"],
    }
