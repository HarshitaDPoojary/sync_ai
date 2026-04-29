from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.graph import ExtractedActionItem, MeetingState

_extraction_chain_cache = None
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def get_extraction_chain():
    global _extraction_chain_cache
    if _extraction_chain_cache is None:
        settings = get_settings()
        prompt_text = (_PROMPTS_DIR / "extraction_system.txt").read_text()
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{full_transcript}"),
        ])
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=settings.groq_extraction_temperature,
            api_key=settings.groq_api_key,
        )
        _extraction_chain_cache = prompt | llm | JsonOutputParser()
    return _extraction_chain_cache


def _verify_quote(quote: str, transcript: str) -> bool:
    return quote.lower().strip() in transcript.lower()


def run_extraction_node(state: MeetingState) -> MeetingState:
    chunks = state["transcript_chunks"]
    if not chunks:
        return state

    full_transcript = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)
    email_map = {p["name"].lower(): p.get("email") for p in state["participants"]}

    chain = get_extraction_chain()
    result = chain.invoke({"full_transcript": full_transcript})

    action_items = []
    for item in result.get("action_items", []):
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
        "action_items": state["action_items"] + action_items,
        "summary": summary,
        "last_checkpoint_chunk_count": len(chunks),
    }
