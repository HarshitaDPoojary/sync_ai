import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.graph import ExtractedActionItem, MeetingState

logger = logging.getLogger("sync_ai.extraction")

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=4)
def get_extraction_chain(model: str, temperature: float):
    prompt_text = (_PROMPTS_DIR / "extraction_system.txt").read_text()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{full_transcript}"),
    ])
    settings = get_settings()
    llm = ChatGroq(model=model, temperature=temperature, api_key=settings.groq_api_key)
    return prompt | llm | JsonOutputParser()


def _verify_quote(quote: str, transcript: str) -> bool:
    return quote.lower().strip() in transcript.lower()


def run_extraction_node(state: MeetingState) -> MeetingState:
    chunks = state["transcript_chunks"]
    if not chunks:
        return state

    full_transcript = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)
    email_map = {p["name"].lower(): p.get("email") for p in state["participants"]}

    settings = get_settings()
    chain = get_extraction_chain(settings.groq_model, settings.groq_extraction_temperature)
    try:
        result = chain.invoke({"full_transcript": full_transcript})
    except Exception as exc:
        logger.warning("extraction_chain_failed error=%s", exc)
        return state

    # Re-extract over the full transcript each checkpoint — replace, don't append,
    # to avoid duplicating items that were already found in an earlier checkpoint.
    action_items = []
    for item in result.get("action_items", []):
        if not item.get("task"):
            logger.debug("extraction_skipped_missing_task item=%s", item)
            continue
        quote = item.get("supporting_quote", "")
        needs_review = not _verify_quote(quote, full_transcript)
        owner_name = item.get("owner_name")
        owner_email = email_map.get(owner_name.lower()) if owner_name else None
        action_items.append(ExtractedActionItem(
            task=item["task"],
            owner_name=owner_name,
            owner_email=owner_email,
            deadline=item.get("deadline"),
            confidence=item.get("confidence", 0.0),
            supporting_quote=quote,
            needs_review=needs_review,
        ))

    summary = result.get("summary", {})
    return {
        **state,
        "action_items": action_items,  # replace, not append — extraction is over full transcript
        "summary": summary,
        "last_checkpoint_chunk_count": len(chunks),
    }
