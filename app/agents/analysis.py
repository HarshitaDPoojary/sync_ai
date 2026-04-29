import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.graph import MeetingState, Signal, Suggestion

logger = logging.getLogger("sync_ai.analysis")

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=4)
def get_analysis_chain(model: str, temperature: float):
    prompt_text = (_PROMPTS_DIR / "analysis_system.txt").read_text()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{transcript_window}"),
    ])
    settings = get_settings()
    llm = ChatGroq(model=model, temperature=temperature, api_key=settings.groq_api_key)
    return prompt | llm | JsonOutputParser()


def run_analysis_node(state: MeetingState) -> MeetingState:
    window = state["analysis_window"]
    if not window:
        return state

    window_text = "\n".join(f"{c['speaker']}: {c['text']}" for c in window)
    settings = get_settings()
    chain = get_analysis_chain(settings.groq_model, settings.groq_analysis_temperature)
    try:
        result = chain.invoke({"transcript_window": window_text})
    except Exception as exc:
        logger.warning("analysis_chain_failed error=%s", exc)
        return state

    new_suggestions = []
    for s in result.get("suggestions", []):
        if not s.get("type") or not s.get("text"):
            logger.debug("analysis_skipped_malformed_suggestion item=%s", s)
            continue
        new_suggestions.append(Suggestion(
            suggestion_type=s["type"],
            text=s["text"],
            confidence=s.get("confidence", 0.8),
        ))

    new_signals = []
    for sig in result.get("signals", []):
        if not sig.get("signal_type") or not sig.get("summary"):
            logger.debug("analysis_skipped_malformed_signal item=%s", sig)
            continue
        new_signals.append(Signal(
            signal_type=sig["signal_type"],
            summary=sig["summary"],
            speaker=sig.get("speaker", "Unknown"),
            timestamp=sig.get("timestamp", 0.0),
        ))

    return {
        **state,
        "suggestions": state["suggestions"] + new_suggestions,
        "signals": state["signals"] + new_signals,
        "last_analysis_token_count": state["token_count"],
    }
