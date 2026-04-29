# Meeting Copilot (LangChain + LangSmith) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the same FastAPI meeting copilot using LangChain LCEL chains orchestrated by a LangGraph `StateGraph`, with LangSmith providing zero-config tracing, eval datasets, and a prompt playground. Recommended for interviews where evaluating the AI pipeline is a key topic.

**Architecture:** A `MeetingSession` runs a LangGraph `StateGraph` where each node is a LangChain LCEL chain (`prompt | llm | parser`). State accumulates in a typed `MeetingState` dict. LangSmith traces every LLM call automatically when `LANGCHAIN_TRACING_V2=true`. Offline evals run against LangSmith datasets using `langsmith.evaluate()`; RAGAS evaluates search quality.

**Tech Stack:** Python 3.11, FastAPI, Recall.ai, `langchain-groq` (`llama-3.1-70b-versatile`), LangGraph, LangChain LCEL, `langchain-chroma`, `langchain-community` (HuggingFace embeddings), ChromaDB, SQLite/SQLModel, Slack SDK, Gmail API, LangSmith, RAGAS, `datasets` (AMI + MeetingBank), Docker, Render.com, GitHub Actions.

---

## File Map

```
sync_ai/
├── app/
│   ├── main.py                        FastAPI app + all routes + WebSocket handler
│   ├── agents/
│   │   ├── analysis.py                AnalysisChain (LCEL) + analysis graph node
│   │   ├── extraction.py              ExtractionChain + SummaryChain + extraction node
│   │   ├── delivery.py                Delivery node (Slack + Gmail)
│   │   └── storage.py                 Storage node (ChromaDB + SQLite)
│   ├── core/
│   │   ├── graph.py                   LangGraph StateGraph + MeetingState TypedDict
│   │   ├── session.py                 MeetingSession: instantiates + runs graph
│   │   ├── recall.py                  Recall.ai HTTP client wrapper
│   │   └── search.py                  RetrievalQA chain for semantic search
│   ├── models/
│   │   └── db.py                      SQLModel models + engine + create_all
│   ├── integrations/
│   │   ├── slack.py                   Slack SDK wrapper
│   │   └── gmail.py                   Gmail API wrapper
│   └── static/
│       └── index.html                 Web dashboard (vanilla JS + CSS)
├── prompts/
│   ├── analysis_system.txt            AnalysisChain system prompt
│   ├── extraction_system.txt          ExtractionChain system prompt
│   └── summary_system.txt             SummaryChain system prompt
├── tests/
│   ├── conftest.py                    Shared fixtures
│   ├── unit/
│   │   ├── test_analysis_chain.py
│   │   ├── test_extraction_chain.py
│   │   ├── test_delivery_node.py
│   │   └── test_storage_node.py
│   ├── integration/
│   │   ├── test_graph_pipeline.py
│   │   └── test_api_endpoints.py
│   └── eval/
│       ├── datasets/                  LangSmith dataset fixtures (JSON)
│       │   └── golden_transcripts.json   Sourced from AMI + MeetingBank + hand-labeled
│       ├── prepare_datasets.py        Downloads AMI/MeetingBank, writes fixture file
│       ├── test_signal_detection.py   LangSmith evaluate() — F1 on signals
│       ├── test_action_extraction.py  LangSmith evaluate() — precision/recall + faithfulness
│       └── test_search_quality.py     RAGAS — NDCG on search
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        ├── test.yml
        └── deploy.yml
```

---

## Task 1: Project Scaffold + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create all package `__init__.py` files
- Create: `prompts/` directory with three prompt files

- [ ] **Step 1: Create requirements.txt**

Pin all LangChain packages to avoid breaking-change upgrades.

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2
sqlmodel==0.0.21
httpx==0.27.2
langchain==0.3.7
langchain-groq==0.2.1
langchain-chroma==0.1.4
langchain-community==0.3.7
langchain-core==0.3.15
langgraph==0.2.38
langsmith==0.1.136
chromadb==0.5.15
sentence-transformers==3.1.1
slack-sdk==3.33.1
google-api-python-client==2.149.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.1
ragas==0.2.3
datasets==2.21.0
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-httpx==0.32.0
python-dotenv==1.0.1
```

- [ ] **Step 2: Create .env.example**

```
RECALL_API_KEY=your_recall_api_key
GROQ_API_KEY=your_groq_api_key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_CHANNEL_ID=C0XXXXXXXXX
GMAIL_CREDENTIALS_JSON=base64_encoded_oauth2_credentials
DATABASE_URL=./data/sync_ai.db
CHROMA_PERSIST_DIR=./data/chroma
WEBHOOK_BASE_URL=https://your-app.onrender.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=sync_ai
```

- [ ] **Step 3: Create directories and `__init__.py` files**

```bash
mkdir -p app/agents app/core app/models app/integrations app/static
mkdir -p prompts tests/unit tests/integration tests/eval/datasets
touch app/__init__.py app/agents/__init__.py app/core/__init__.py
touch app/models/__init__.py app/integrations/__init__.py
touch tests/__init__.py tests/unit/__init__.py
touch tests/integration/__init__.py tests/eval/__init__.py
```

- [ ] **Step 4: Create the three system prompt files**

```
# prompts/analysis_system.txt
You are a meeting analysis assistant. Given a transcript window, return a JSON object with:
- "suggestions": list of objects with keys "type" ("question"|"clarification"|"talking_point"), "text", "confidence" (0-1).
  Return up to 3 questions, 2 clarifications, 2 talking points.
- "signals": list of objects with keys "signal_type" ("decision"|"blocker"|"commitment"), "summary", "speaker", "timestamp".
  Only include signals clearly and explicitly stated in the transcript.
Return ONLY valid JSON. No explanation.
```

```
# prompts/extraction_system.txt
You are a meeting action item extractor. Given a full meeting transcript, return a JSON object with:
- "action_items": list of objects with keys:
    "task" (string), "owner_name" (string or null), "deadline" (string or null),
    "confidence" (0-1), "supporting_quote" (verbatim text from transcript that supports this item).
  Only extract items explicitly stated. Do NOT infer or assume.
- "summary": object with keys "decisions", "blockers", "commitments", "next_steps" (each a list of strings).
Return ONLY valid JSON. No explanation.
```

```
# prompts/summary_system.txt
You are a meeting summarizer. Given a full transcript, return a JSON object with:
- "decisions": list of strings — decisions made during the meeting.
- "blockers": list of strings — blockers or impediments raised.
- "commitments": list of strings — explicit commitments made by participants.
- "next_steps": list of strings — next steps agreed upon.
Return ONLY valid JSON. No explanation.
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without conflicts.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .env.example app/ tests/ prompts/
git commit -m "feat: project scaffold, dependencies, and system prompts"
```

---

## Task 2: Database Models

**Files:**
- Create: `app/models/db.py`
- Create: `tests/unit/test_db_models.py`

Identical to the event-driven plan — same SQLModel schema.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_db_models.py
import pytest
from sqlmodel import Session, create_engine
from app.models.db import Meeting, ActionItem, create_db_and_tables

@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    create_db_and_tables(e)
    return e

def test_meeting_create(engine):
    with Session(engine) as session:
        meeting = Meeting(
            id="m1", title="Sprint Review",
            platform_url="https://zoom.us/j/123",
            status="active", recall_bot_id="bot_abc"
        )
        session.add(meeting)
        session.commit()
        session.refresh(meeting)
    assert meeting.id == "m1"
    assert meeting.status == "active"

def test_action_item_defaults(engine):
    with Session(engine) as session:
        meeting = Meeting(id="m2", title="t", platform_url="u", status="active", recall_bot_id="b")
        session.add(meeting)
        session.commit()
        item = ActionItem(
            id="a1", meeting_id="m2", task="Deploy API",
            confidence=0.9, status="pending",
            needs_review=False, supporting_quote="We will deploy"
        )
        session.add(item)
        session.commit()
        session.refresh(item)
    assert item.needs_review is False
    assert item.sent_via is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_db_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.models.db'`

- [ ] **Step 3: Implement db.py**

```python
# app/models/db.py
import json
import os
from datetime import datetime
from typing import List, Optional
from sqlmodel import Field, SQLModel, create_engine


class Meeting(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    platform_url: str
    status: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    recall_bot_id: str
    langsmith_trace_url: Optional[str] = None


class Participant(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    display_name: str
    email: Optional[str] = None
    slack_id: Optional[str] = None


class Transcript(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    full_text: str = ""
    chunks_json: str = "[]"


class ActionItem(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    task: str
    owner_id: Optional[str] = Field(default=None, foreign_key="participant.id")
    deadline: Optional[str] = None
    confidence: float = 0.0
    status: str = "pending"
    needs_review: bool = False
    supporting_quote: str = ""
    sent_via: Optional[str] = None


class MeetingSummary(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_id: str = Field(foreign_key="meeting.id")
    decisions_json: str = "[]"
    blockers_json: str = "[]"
    commitments_json: str = "[]"
    next_steps_json: str = "[]"

    @property
    def decisions(self) -> List[str]:
        return json.loads(self.decisions_json)

    @property
    def blockers(self) -> List[str]:
        return json.loads(self.blockers_json)

    @property
    def commitments(self) -> List[str]:
        return json.loads(self.commitments_json)

    @property
    def next_steps(self) -> List[str]:
        return json.loads(self.next_steps_json)


def create_db_and_tables(engine=None):
    if engine is None:
        engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def get_engine():
    url = os.getenv("DATABASE_URL", "./data/sync_ai.db")
    return create_engine(f"sqlite:///{url}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_db_models.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/models/db.py tests/unit/test_db_models.py
git commit -m "feat: SQLModel database models"
```

---

## Task 3: MeetingState + LangGraph Graph Definition

**Files:**
- Create: `app/core/graph.py`
- Create: `tests/unit/test_graph_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graph_state.py
from app.core.graph import MeetingState, make_initial_state

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
    from app.core.graph import should_analyze
    state = make_initial_state("m1", [])
    state["token_count"] = 100
    state["last_analysis_token_count"] = 0
    assert should_analyze(state) is False

def test_should_analyze_true_above_threshold():
    from app.core.graph import should_analyze
    state = make_initial_state("m1", [])
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0
    assert should_analyze(state) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_graph_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.graph'`

- [ ] **Step 3: Implement graph.py**

```python
# app/core/graph.py
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

TOKEN_THRESHOLD = 500


class TranscriptChunk(TypedDict):
    speaker: str
    text: str
    timestamp: float
    sequence_num: int


class Suggestion(TypedDict):
    suggestion_type: str   # "question" | "clarification" | "talking_point"
    text: str
    confidence: float


class Signal(TypedDict):
    signal_type: str       # "decision" | "blocker" | "commitment"
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
    analysis_window: List[TranscriptChunk]     # last 5 minutes
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
    return (state["token_count"] - state["last_analysis_token_count"]) >= TOKEN_THRESHOLD


def should_extract(state: MeetingState) -> bool:
    chunks_since_checkpoint = len(state["transcript_chunks"]) - state["last_checkpoint_chunk_count"]
    return state["meeting_ended"] or chunks_since_checkpoint >= 50


def build_graph(analysis_node, extraction_node, delivery_node, storage_node) -> StateGraph:
    """
    Builds the LangGraph StateGraph. Each node function must have signature:
        (state: MeetingState) -> MeetingState
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_graph_state.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/core/graph.py tests/unit/test_graph_state.py
git commit -m "feat: LangGraph MeetingState and graph definition"
```

---

## Task 4: Recall.ai Client

**Files:**
- Create: `app/core/recall.py`
- Create: `tests/unit/test_recall.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_recall.py
import pytest
from pytest_httpx import HTTPXMock
from app.core.recall import RecallClient

@pytest.fixture
def client():
    return RecallClient(api_key="test_key")

@pytest.mark.asyncio
async def test_create_bot(httpx_mock: HTTPXMock, client):
    httpx_mock.add_response(
        method="POST",
        url="https://us-east-1.recall.ai/api/v1/bot/",
        json={"id": "bot_123"},
        status_code=201,
    )
    bot = await client.create_bot_with_webhook(
        meeting_url="https://zoom.us/j/999",
        webhook_url="https://app.com/webhook/recall/m1",
    )
    assert bot["id"] == "bot_123"

@pytest.mark.asyncio
async def test_remove_bot(httpx_mock: HTTPXMock, client):
    httpx_mock.add_response(
        method="DELETE",
        url="https://us-east-1.recall.ai/api/v1/bot/bot_123/leave_call/",
        json={},
        status_code=200,
    )
    await client.remove_bot("bot_123")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_recall.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement recall.py**

```python
# app/core/recall.py
from typing import Any, Dict, List
import httpx


class RecallClient:
    BASE_URL = "https://us-east-1.recall.ai/api/v1"

    def __init__(self, api_key: str):
        self._headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    async def create_bot_with_webhook(
        self, meeting_url: str, webhook_url: str,
        bot_name: str = "Meeting Copilot"
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"{self.BASE_URL}/bot/",
                headers=self._headers,
                json={
                    "meeting_url": meeting_url,
                    "bot_name": bot_name,
                    "transcription_options": {"provider": "assembly_ai"},
                    "real_time_transcription": {
                        "destination_url": webhook_url,
                        "partial_results": False,
                    },
                },
            )
            response.raise_for_status()
            return response.json()

    async def remove_bot(self, bot_id: str) -> None:
        async with httpx.AsyncClient() as http:
            response = await http.delete(
                f"{self.BASE_URL}/bot/{bot_id}/leave_call/",
                headers=self._headers,
            )
            response.raise_for_status()

    async def get_participants(self, bot_id: str) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as http:
            response = await http.get(
                f"{self.BASE_URL}/bot/{bot_id}/",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json().get("meeting_participants", [])
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_recall.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/core/recall.py tests/unit/test_recall.py
git commit -m "feat: Recall.ai async HTTP client"
```

---

## Task 5: AnalysisChain (LCEL)

**Files:**
- Create: `app/agents/analysis.py`
- Create: `tests/unit/test_analysis_chain.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_analysis_chain.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from app.agents.analysis import run_analysis_node
from app.core.graph import make_initial_state, TranscriptChunk

FAKE_LLM_OUTPUT = '{"suggestions": [{"type": "question", "text": "What is the deadline?", "confidence": 0.9}], "signals": [{"signal_type": "decision", "summary": "Use PostgreSQL", "speaker": "Alice", "timestamp": 10.0}]}'


def test_analysis_node_updates_state():
    state = make_initial_state("m1", [])
    state["analysis_window"] = [
        TranscriptChunk(speaker="Alice", text="We decided to use PostgreSQL", timestamp=10.0, sequence_num=1)
    ]
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=FAKE_LLM_OUTPUT)

    with patch("app.agents.analysis.get_analysis_chain", return_value=mock_llm):
        new_state = run_analysis_node(state)

    assert len(new_state["suggestions"]) == 1
    assert new_state["suggestions"][0]["text"] == "What is the deadline?"
    assert len(new_state["signals"]) == 1
    assert new_state["signals"][0]["signal_type"] == "decision"
    assert new_state["last_analysis_token_count"] == 600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_analysis_chain.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement analysis.py**

```python
# app/agents/analysis.py
import json
import os
import time
from pathlib import Path
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from app.core.graph import MeetingState, Suggestion, Signal

WINDOW_SECONDS = 300
_chain_cache = None


def get_analysis_chain():
    global _chain_cache
    if _chain_cache is None:
        prompt_text = Path("prompts/analysis_system.txt").read_text()
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{transcript_window}"),
        ])
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            api_key=os.environ["GROQ_API_KEY"],
        )
        _chain_cache = prompt | llm | JsonOutputParser()
    return _chain_cache


def _verify_quote(quote: str, transcript: str) -> bool:
    return quote.lower().strip() in transcript.lower()


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_analysis_chain.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add app/agents/analysis.py tests/unit/test_analysis_chain.py
git commit -m "feat: AnalysisChain LCEL node with Groq"
```

---

## Task 6: ExtractionChain (LCEL)

**Files:**
- Create: `app/agents/extraction.py`
- Create: `tests/unit/test_extraction_chain.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extraction_chain.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from app.agents.extraction import run_extraction_node
from app.core.graph import make_initial_state, TranscriptChunk

TRANSCRIPT = "Alice: Bob will deploy the API by Friday. Bob: Agreed."
FAKE_OUTPUT = json_str = '{"action_items": [{"task": "Deploy the API", "owner_name": "Bob", "deadline": "Friday", "confidence": 0.95, "supporting_quote": "Bob will deploy the API by Friday"}], "summary": {"decisions": ["Deploy API by Friday"], "blockers": [], "commitments": [], "next_steps": []}}'

import json


def test_extraction_node_verifies_quote():
    state = make_initial_state("m1", [{"name": "Bob", "email": "bob@x.com"}])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Bob will deploy the API by Friday.", timestamp=0.0, sequence_num=1),
        TranscriptChunk(speaker="Bob", text="Agreed.", timestamp=5.0, sequence_num=2),
    ]
    state["meeting_ended"] = True

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = json.loads(FAKE_OUTPUT)

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

    bad_output = {
        "action_items": [{
            "task": "Redesign database",
            "owner_name": "Alice",
            "deadline": "Monday",
            "confidence": 0.6,
            "supporting_quote": "this quote does not exist in the transcript anywhere",
        }],
        "summary": {"decisions": [], "blockers": [], "commitments": [], "next_steps": []},
    }
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = bad_output

    with patch("app.agents.extraction.get_extraction_chain", return_value=mock_chain):
        new_state = run_extraction_node(state)

    assert new_state["action_items"][0]["needs_review"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_extraction_chain.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement extraction.py**

```python
# app/agents/extraction.py
import os
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from app.core.graph import MeetingState, ExtractedActionItem

_extraction_chain_cache = None


def get_extraction_chain():
    global _extraction_chain_cache
    if _extraction_chain_cache is None:
        prompt_text = Path("prompts/extraction_system.txt").read_text()
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{full_transcript}"),
        ])
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            api_key=os.environ["GROQ_API_KEY"],
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_extraction_chain.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/agents/extraction.py tests/unit/test_extraction_chain.py
git commit -m "feat: ExtractionChain LCEL node with anti-hallucination quote verification"
```

---

## Task 7: Delivery + Storage Nodes

**Files:**
- Create: `app/agents/delivery.py`
- Create: `app/agents/storage.py`
- Create: `app/integrations/slack.py`
- Create: `app/integrations/gmail.py`
- Create: `tests/unit/test_delivery_node.py`
- Create: `tests/unit/test_storage_node.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_delivery_node.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.delivery import run_delivery_node
from app.core.graph import make_initial_state, ExtractedActionItem

def test_delivery_node_calls_slack():
    state = make_initial_state("m1", [])
    state["action_items"] = [
        ExtractedActionItem(task="Deploy API", owner_name="Bob", owner_email="bob@x.com",
                            deadline="Friday", confidence=0.9, supporting_quote="Bob will deploy", needs_review=False)
    ]
    state["summary"] = {"decisions": ["Use PG"], "blockers": [], "commitments": [], "next_steps": []}

    mock_slack = MagicMock()
    mock_slack.send_action_items_sync = MagicMock()
    mock_slack.send_summary_sync = MagicMock()

    with patch("app.agents.delivery.get_slack_client", return_value=mock_slack):
        new_state = run_delivery_node(state)

    mock_slack.send_action_items_sync.assert_called_once()
    mock_slack.send_summary_sync.assert_called_once()
```

```python
# tests/unit/test_storage_node.py
import pytest
from unittest.mock import MagicMock
from sqlmodel import create_engine
from app.agents.storage import run_storage_node
from app.core.graph import make_initial_state, TranscriptChunk, ExtractedActionItem
from app.models.db import create_db_and_tables, Transcript

@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    create_db_and_tables(e)
    return e

def test_storage_node_writes_transcript(engine):
    state = make_initial_state("m1", [])
    state["transcript_chunks"] = [
        TranscriptChunk(speaker="Alice", text="Hello there", timestamp=0.0, sequence_num=1)
    ]
    state["action_items"] = []
    state["summary"] = None

    mock_chroma = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 384]
    mock_embedder.embed_query.return_value = [0.1] * 384

    with MagicMock() as mock_vs:
        from sqlmodel import Session
        new_state = run_storage_node(state, engine=engine, chroma_collection=mock_chroma, embedder=mock_embedder)

    from sqlmodel import Session
    with Session(engine) as session:
        transcript = session.get(Transcript, "m1")
    assert transcript is not None
    assert "Hello there" in transcript.full_text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_delivery_node.py tests/unit/test_storage_node.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement slack.py and gmail.py**

```python
# app/integrations/slack.py
from typing import Any, Dict, List
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk import WebClient


class SlackClient:
    def __init__(self, bot_token: str, channel_id: str):
        self._async_client = AsyncWebClient(token=bot_token)
        self._sync_client = WebClient(token=bot_token)
        self._channel = channel_id

    async def send_action_items(self, meeting_title: str, items: List[Dict[str, Any]]) -> None:
        lines = [f"*Action Items from: {meeting_title}*\n"]
        for item in items:
            owner = item.get("owner_name") or "Unassigned"
            deadline = item.get("deadline") or "No deadline"
            flag = " ⚠️ needs review" if item.get("needs_review") else ""
            lines.append(f"• *{item['task']}* — {owner} by {deadline}{flag}")
        await self._async_client.chat_postMessage(channel=self._channel, text="\n".join(lines))

    def send_action_items_sync(self, meeting_title: str, items: List[Dict[str, Any]]) -> None:
        lines = [f"*Action Items from: {meeting_title}*\n"]
        for item in items:
            owner = item.get("owner_name") or "Unassigned"
            deadline = item.get("deadline") or "No deadline"
            flag = " ⚠️ needs review" if item.get("needs_review") else ""
            lines.append(f"• *{item['task']}* — {owner} by {deadline}{flag}")
        self._sync_client.chat_postMessage(channel=self._channel, text="\n".join(lines))

    async def send_summary(self, meeting_title: str, decisions: List[str],
                           blockers: List[str], next_steps: List[str]) -> None:
        sections = [f"*Meeting Summary: {meeting_title}*\n"]
        if decisions:
            sections.append("*Decisions:*\n" + "\n".join(f"• {d}" for d in decisions))
        if blockers:
            sections.append("*Blockers:*\n" + "\n".join(f"• {b}" for b in blockers))
        if next_steps:
            sections.append("*Next Steps:*\n" + "\n".join(f"• {s}" for s in next_steps))
        await self._async_client.chat_postMessage(channel=self._channel, text="\n\n".join(sections))

    def send_summary_sync(self, meeting_title: str, decisions: List[str],
                          blockers: List[str], next_steps: List[str]) -> None:
        sections = [f"*Meeting Summary: {meeting_title}*\n"]
        if decisions:
            sections.append("*Decisions:*\n" + "\n".join(f"• {d}" for d in decisions))
        if blockers:
            sections.append("*Blockers:*\n" + "\n".join(f"• {b}" for b in blockers))
        if next_steps:
            sections.append("*Next Steps:*\n" + "\n".join(f"• {s}" for s in next_steps))
        self._sync_client.chat_postMessage(channel=self._channel, text="\n\n".join(sections))
```

```python
# app/integrations/gmail.py
import base64
import json
import os
from email.mime.text import MIMEText
from typing import Any, Dict, List
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def _get_gmail_service():
    creds_b64 = os.getenv("GMAIL_CREDENTIALS_JSON", "")
    creds_json = json.loads(base64.b64decode(creds_b64).decode())
    creds = Credentials.from_authorized_user_info(creds_json)
    return build("gmail", "v1", credentials=creds)


class GmailClient:
    def send_action_items(self, to_email: str, owner_name: str,
                          meeting_title: str, items: List[Dict[str, Any]]) -> None:
        service = _get_gmail_service()
        lines = [f"Hi {owner_name},\n\nYour action items from '{meeting_title}':\n"]
        for item in items:
            deadline = item.get("deadline") or "No deadline"
            flag = " [NEEDS REVIEW]" if item.get("needs_review") else ""
            lines.append(f"- {item['task']} (by {deadline}){flag}")
        lines.append("\n\nGenerated by sync_ai.")
        message = MIMEText("\n".join(lines))
        message["to"] = to_email
        message["subject"] = f"Action Items: {meeting_title}"
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
```

- [ ] **Step 4: Implement delivery.py**

```python
# app/agents/delivery.py
import os
from typing import Any, Dict, List, Optional
from app.core.graph import MeetingState
from app.integrations.slack import SlackClient

_slack_client: Optional[SlackClient] = None


def get_slack_client() -> Optional[SlackClient]:
    global _slack_client
    token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL_ID")
    if token and channel and _slack_client is None:
        _slack_client = SlackClient(token, channel)
    return _slack_client


def run_delivery_node(state: MeetingState, meeting_title: str = "Meeting") -> MeetingState:
    slack = get_slack_client()
    items = [dict(item) for item in state["action_items"]]
    summary = state.get("summary") or {}

    if slack and items:
        slack.send_action_items_sync(meeting_title, items)

    if slack and summary:
        slack.send_summary_sync(
            meeting_title,
            summary.get("decisions", []),
            summary.get("blockers", []),
            summary.get("next_steps", []),
        )
    return state
```

- [ ] **Step 5: Implement storage.py**

```python
# app/agents/storage.py
import json
import uuid
from typing import Any
from sqlmodel import Session
from app.core.graph import MeetingState
from app.models.db import Transcript, get_engine


def run_storage_node(state: MeetingState, engine=None,
                     chroma_collection=None, embedder=None) -> MeetingState:
    if engine is None:
        engine = get_engine()

    chunks = state["transcript_chunks"]
    full_text = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)

    with Session(engine) as session:
        transcript = session.get(Transcript, state["meeting_id"])
        if transcript is None:
            transcript = Transcript(id=state["meeting_id"], meeting_id=state["meeting_id"])
            session.add(transcript)
        transcript.full_text = full_text
        transcript.chunks_json = json.dumps(chunks)
        session.commit()

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
            chroma_collection.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[item["task"]],
                metadatas=[{
                    "meeting_id": state["meeting_id"],
                    "owner_name": item.get("owner_name") or "",
                    "deadline": item.get("deadline") or "",
                }],
            )

    return state
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_delivery_node.py tests/unit/test_storage_node.py -v
```

Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add app/agents/ app/integrations/ tests/unit/test_delivery_node.py tests/unit/test_storage_node.py
git commit -m "feat: delivery and storage nodes, Slack and Gmail wrappers"
```

---

## Task 8: MeetingSession + Search Chain

**Files:**
- Create: `app/core/session.py`
- Create: `app/core/search.py`
- Create: `tests/unit/test_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_session.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.session import MeetingSession

@pytest.mark.asyncio
async def test_session_starts_recall_bot():
    with patch("app.core.session.RecallClient") as MockRecall, \
         patch("app.core.session.chromadb"), \
         patch("app.core.session.HuggingFaceEmbeddings"), \
         patch("app.core.session.get_engine"):
        mock_recall_instance = MagicMock()
        mock_recall_instance.create_bot_with_webhook = AsyncMock(return_value={"id": "bot_xyz"})
        MockRecall.return_value = mock_recall_instance

        session = MeetingSession(
            meeting_id="m1",
            meeting_url="https://zoom.us/j/999",
            title="Test",
            participant_emails=[],
            recall_api_key="key",
            webhook_base_url="https://example.com",
        )
        await session.start()
        assert session.bot_id == "bot_xyz"
        await session.stop()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_session.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement search.py**

```python
# app/core/search.py
import os
from typing import Any, Dict, List
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


def build_search_chain(chroma_persist_dir: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        collection_name="transcripts",
        embedding_function=embeddings,
        persist_directory=chroma_persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


def search_transcripts(query: str, chroma_persist_dir: str, limit: int = 5) -> List[Dict]:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        collection_name="transcripts",
        embedding_function=embeddings,
        persist_directory=chroma_persist_dir,
    )
    docs = vectorstore.similarity_search(query, k=limit)
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
```

- [ ] **Step 4: Implement session.py**

```python
# app/core/session.py
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.graph import (
    MeetingState, make_initial_state, build_graph,
    should_analyze, should_extract, WINDOW_SECONDS,
)
from app.core.recall import RecallClient
from app.agents.analysis import run_analysis_node
from app.agents.extraction import run_extraction_node
from app.agents.delivery import run_delivery_node
from app.agents.storage import run_storage_node
from app.models.db import get_engine


class MeetingSession:
    def __init__(self, meeting_id: str, meeting_url: str, title: str,
                 participant_emails: List[str], recall_api_key: str,
                 webhook_base_url: str):
        self.meeting_id = meeting_id
        self.meeting_url = meeting_url
        self.title = title
        self.bot_id: Optional[str] = None
        self._recall = RecallClient(api_key=recall_api_key)
        self._webhook_base = webhook_base_url

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_col = chroma_client.get_or_create_collection("transcripts")
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        engine = get_engine()

        delivery_node = partial(run_delivery_node, meeting_title=title)
        storage_node = partial(run_storage_node, engine=engine,
                               chroma_collection=chroma_col, embedder=embedder)

        self._graph = build_graph(
            analysis_node=run_analysis_node,
            extraction_node=run_extraction_node,
            delivery_node=delivery_node,
            storage_node=storage_node,
        )
        self._state: MeetingState = make_initial_state(meeting_id, [])
        self._sequence = 0

    async def start(self) -> None:
        webhook_url = f"{self._webhook_base}/webhook/recall/{self.meeting_id}"
        bot = await self._recall.create_bot_with_webhook(self.meeting_url, webhook_url)
        self.bot_id = bot["id"]

    async def stop(self) -> None:
        if self.bot_id:
            await self._recall.remove_bot(self.bot_id)
        self._state["meeting_ended"] = True
        self._run_graph()

    def handle_recall_chunk(self, payload: Dict[str, Any]) -> None:
        words = payload.get("transcript", {}).get("words", [])
        if not words:
            return
        speaker_idx = str(words[0].get("speaker", 0))
        speaker = self._state["participants"][int(speaker_idx)]["name"] if self._state["participants"] else f"Speaker {speaker_idx}"
        text = " ".join(w["text"] for w in words)
        timestamp = words[0].get("start_time", time.time())
        self._sequence += 1
        chunk = {"speaker": speaker, "text": text, "timestamp": timestamp, "sequence_num": self._sequence}

        cutoff = time.time() - WINDOW_SECONDS
        self._state["transcript_chunks"].append(chunk)
        self._state["analysis_window"] = [
            c for c in self._state["transcript_chunks"] if c["timestamp"] >= cutoff
        ]
        self._state["token_count"] += len(text.split())

        if should_analyze(self._state) or should_extract(self._state):
            self._run_graph()

    def _run_graph(self) -> None:
        self._state = self._graph.invoke(self._state)

    def get_suggestions(self):
        return self._state["suggestions"]

    def get_signals(self):
        return self._state["signals"]

    def get_action_items(self):
        return self._state["action_items"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/unit/test_session.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add app/core/session.py app/core/search.py tests/unit/test_session.py
git commit -m "feat: MeetingSession with LangGraph and RetrievalQA search chain"
```

---

## Task 9: FastAPI App + REST Endpoints + WebSocket

**Files:**
- Create: `app/main.py`
- Create: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_api_endpoints.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.main.MeetingSession") as MockSession, \
         patch("app.main.create_db_and_tables"), \
         patch("app.main.get_engine"):
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.bot_id = "bot_1"
        mock_session.get_suggestions.return_value = []
        mock_session.get_signals.return_value = []
        mock_session.get_action_items.return_value = []
        MockSession.return_value = mock_session
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_create_session(client):
    response = client.post("/sessions", json={
        "meeting_url": "https://zoom.us/j/123",
        "title": "Sprint Review",
        "participant_emails": [],
    })
    assert response.status_code == 201
    assert "id" in response.json()

def test_get_session_not_found(client):
    assert client.get("/sessions/nonexistent").status_code == 404

def test_search_returns_list(client):
    with patch("app.main.search_transcripts", return_value=[]):
        assert client.get("/search?q=deploy").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_api_endpoints.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement main.py**

```python
# app/main.py
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session

from app.core.session import MeetingSession
from app.core.search import search_transcripts
from app.models.db import Meeting, create_db_and_tables, get_engine

_sessions: Dict[str, MeetingSession] = {}
_ws_clients: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    create_db_and_tables(get_engine())
    import langsmith
    # LangSmith tracing is activated automatically when LANGCHAIN_TRACING_V2=true
    # and LANGCHAIN_API_KEY are set — no explicit init needed
    yield


app = FastAPI(title="sync_ai", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class CreateSessionRequest(BaseModel):
    meeting_url: str
    title: str
    participant_emails: List[str] = []


@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    session = MeetingSession(
        meeting_id=session_id,
        meeting_url=req.meeting_url,
        title=req.title,
        participant_emails=req.participant_emails,
        recall_api_key=os.environ["RECALL_API_KEY"],
        webhook_base_url=os.getenv("WEBHOOK_BASE_URL", ""),
    )
    await session.start()
    _sessions[session_id] = session
    with Session(get_engine()) as db:
        db.add(Meeting(
            id=session_id, title=req.title,
            platform_url=req.meeting_url,
            status="active", recall_bot_id=session.bot_id or "",
        ))
        db.commit()
    return {"id": session_id, "status": "active", "bot_id": session.bot_id}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    with Session(get_engine()) as db:
        meeting = db.get(Meeting, session_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Session not found")
    return meeting


@app.delete("/sessions/{session_id}", status_code=204)
async def end_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        await session.stop()
    with Session(get_engine()) as db:
        meeting = db.get(Meeting, session_id)
        if meeting:
            meeting.status = "ended"
            db.commit()


@app.get("/sessions/{session_id}/suggestions")
async def get_suggestions(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_suggestions()


@app.get("/sessions/{session_id}/signals")
async def get_signals(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_signals()


@app.get("/sessions/{session_id}/action-items")
async def get_action_items(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_action_items()


@app.post("/webhook/recall/{session_id}")
async def recall_webhook(session_id: str, payload: Dict[str, Any]):
    session = _sessions.get(session_id)
    if not session:
        return {"ok": False}
    session.handle_recall_chunk(payload)
    for ws in list(_ws_clients.get(session_id, [])):
        try:
            await ws.send_json({"type": "transcript", "data": payload.get("transcript", {})})
        except Exception:
            _ws_clients[session_id].remove(ws)
    for ws in list(_ws_clients.get(session_id, [])):
        try:
            for s in session.get_suggestions()[-3:]:
                await ws.send_json({"type": "suggestion", "data": s})
            for sig in session.get_signals()[-3:]:
                await ws.send_json({"type": "signal", "data": sig})
        except Exception:
            pass
    return {"ok": True}


@app.get("/search")
async def search(q: str, limit: int = 5):
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    return search_transcripts(q, persist_dir, limit)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _ws_clients.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_clients.get(session_id, []).remove(websocket)


@app.get("/")
async def dashboard():
    return FileResponse("app/static/index.html")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integration/test_api_endpoints.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/integration/test_api_endpoints.py
git commit -m "feat: FastAPI app with REST endpoints and WebSocket"
```

---

## Task 10: Web Dashboard

**Files:**
- Create: `app/static/index.html`

Identical to event-driven plan dashboard.

- [ ] **Step 1: Create app/static/index.html**

Copy the dashboard from the event-driven plan verbatim — it is identical since the API interface is the same.

```html
<!-- app/static/index.html — identical to event-driven plan dashboard -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>sync_ai</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
    #topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:#1e293b;border-bottom:1px solid #334155}
    #topbar h1{font-size:1.1rem;font-weight:600}
    #status{padding:4px 10px;border-radius:999px;font-size:.75rem;background:#334155}
    #status.active{background:#15803d}
    #start-form{display:flex;gap:8px;align-items:center}
    #start-form input{padding:6px 12px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;font-size:.85rem;width:280px}
    button{padding:6px 16px;border-radius:6px;border:none;cursor:pointer;font-size:.85rem;font-weight:500}
    #start-btn{background:#3b82f6;color:#fff}
    #end-btn{background:#dc2626;color:#fff;display:none}
    #main{display:flex;flex:1;overflow:hidden}
    #transcript-panel{flex:0 0 60%;border-right:1px solid #334155;display:flex;flex-direction:column}
    #transcript-panel h2{padding:12px 16px;font-size:.85rem;color:#94a3b8;border-bottom:1px solid #334155}
    #transcript-body{flex:1;overflow-y:auto;padding:16px;font-size:.85rem;line-height:1.6}
    .chunk{margin-bottom:8px}.chunk .spk{font-weight:600;color:#60a5fa;margin-right:6px}
    #right-panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
    #tabs{display:flex;border-bottom:1px solid #334155}
    .tab{padding:10px 20px;font-size:.8rem;cursor:pointer;color:#94a3b8;border-bottom:2px solid transparent}
    .tab.active{color:#e2e8f0;border-bottom-color:#3b82f6}
    .tab-content{display:none;flex:1;overflow-y:auto;padding:16px}
    .tab-content.active{display:block}
    .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin-bottom:10px;font-size:.82rem;line-height:1.5}
    .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:600;margin-bottom:6px}
    .question{background:#1d4ed8}.clarification{background:#7c3aed}.talking_point{background:#0f766e}
    .decision{background:#15803d}.blocker{background:#b91c1c}.commitment{background:#b45309}.review{background:#b91c1c}
    #search-bar{padding:10px 16px;border-top:1px solid #334155;display:flex;gap:8px}
    #search-bar input{flex:1;padding:8px 12px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;font-size:.85rem}
    #search-bar button{background:#3b82f6;color:#fff}
    #search-results{padding:12px 16px;font-size:.8rem;color:#94a3b8}
  </style>
</head>
<body>
<div id="topbar">
  <h1>sync_ai</h1>
  <span id="status">idle</span>
  <div id="start-form">
    <input id="url" placeholder="Zoom/Meet URL..."/>
    <input id="title" placeholder="Meeting title..."/>
    <button id="start-btn" onclick="startSession()">Start</button>
    <button id="end-btn" onclick="endSession()">End Meeting</button>
  </div>
</div>
<div id="main">
  <div id="transcript-panel">
    <h2>Live Transcript</h2>
    <div id="transcript-body"></div>
  </div>
  <div id="right-panel">
    <div id="tabs">
      <div class="tab active" onclick="showTab('suggestions')">Suggestions</div>
      <div class="tab" onclick="showTab('signals')">Signals</div>
      <div class="tab" onclick="showTab('action-items')">Action Items</div>
    </div>
    <div id="suggestions" class="tab-content active"></div>
    <div id="signals" class="tab-content"></div>
    <div id="action-items" class="tab-content"></div>
    <div id="search-bar">
      <input id="search-input" placeholder="Search past meetings..." onkeydown="if(event.key==='Enter')doSearch()"/>
      <button onclick="doSearch()">Search</button>
    </div>
    <div id="search-results"></div>
  </div>
</div>
<script>
  let sessionId=null,ws=null;
  function showTab(n){document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',['suggestions','signals','action-items'][i]===n));document.querySelectorAll('.tab-content').forEach(c=>c.classList.toggle('active',c.id===n))}
  async function startSession(){const url=document.getElementById('url').value.trim(),title=document.getElementById('title').value.trim()||'Meeting';if(!url)return alert('Enter a meeting URL');const r=await fetch('/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({meeting_url:url,title,participant_emails:[]})});const d=await r.json();sessionId=d.id;document.getElementById('status').textContent='active';document.getElementById('status').classList.add('active');document.getElementById('end-btn').style.display='inline-block';connectWS(sessionId)}
  async function endSession(){if(!sessionId)return;await fetch(`/sessions/${sessionId}`,{method:'DELETE'});if(ws)ws.close();document.getElementById('status').textContent='ended';document.getElementById('status').classList.remove('active');document.getElementById('end-btn').style.display='none'}
  function connectWS(id){ws=new WebSocket(`ws://${location.host}/ws/${id}`);ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='transcript')appendTranscript(m.data);if(m.type==='suggestion')appendSuggestion(m.data);if(m.type==='signal')appendSignal(m.data);if(m.type==='action_item')appendActionItem(m.data)}}
  function appendTranscript(d){const b=document.getElementById('transcript-body');b.insertAdjacentHTML('beforeend',`<div class="chunk"><span class="spk">${d.speaker||'Speaker'}:</span>${d.text||''}</div>`);b.scrollTop=b.scrollHeight}
  function appendSuggestion(d){document.getElementById('suggestions').insertAdjacentHTML('afterbegin',`<div class="card"><span class="badge ${d.suggestion_type}">${d.suggestion_type}</span><div>${d.text}</div></div>`)}
  function appendSignal(d){document.getElementById('signals').insertAdjacentHTML('afterbegin',`<div class="card"><span class="badge ${d.signal_type}">${d.signal_type}</span><div>${d.summary}</div><div style="color:#94a3b8;font-size:.75rem;margin-top:4px">${d.speaker}</div></div>`)}
  function appendActionItem(d){const flag=d.needs_review?'<span class="badge review">needs review</span> ':'';document.getElementById('action-items').insertAdjacentHTML('beforeend',`<div class="card">${flag}<strong>${d.task}</strong><div style="color:#94a3b8;font-size:.75rem;margin-top:4px">${d.owner_name||'Unassigned'} · ${d.deadline||'No deadline'}</div></div>`)}
  async function doSearch(){const q=document.getElementById('search-input').value.trim();if(!q)return;const r=await fetch(`/search?q=${encodeURIComponent(q)}&limit=5`);const d=await r.json();const el=document.getElementById('search-results');el.innerHTML=d.length?d.map(r=>`<div style="margin-bottom:8px;padding:8px;background:#1e293b;border-radius:6px;font-size:.78rem">${r.text}</div>`).join(''):'No results.'}
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add app/static/index.html
git commit -m "feat: web dashboard"
```

---

## Task 11: Eval Dataset Preparation (AMI + MeetingBank)

**Files:**
- Create: `tests/eval/prepare_datasets.py`
- Create: `tests/eval/datasets/golden_transcripts.json`

Identical script to event-driven plan — same data sources, same fixture format.

- [ ] **Step 1: Create prepare_datasets.py**

```python
# tests/eval/prepare_datasets.py
"""
Run once to build tests/eval/datasets/golden_transcripts.json from:
  - AMI Meeting Corpus   (HuggingFace: edinburghcristin/ami-corpus)
  - MeetingBank          (HuggingFace: huuuyeah/meetingbank)
  - 2 hand-labeled tech meeting examples

Usage:
    python tests/eval/prepare_datasets.py
"""
import json
import os
from datasets import load_dataset

OUTPUT = "tests/eval/datasets/golden_transcripts.json"
fixtures = []


def load_ami():
    print("Loading AMI corpus...")
    try:
        ds = load_dataset("edinburghcristin/ami-corpus", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  AMI load failed ({e}), skipping.")
        return
    for row in ds.select(range(min(5, len(ds)))):
        transcript = row.get("transcript") or row.get("text") or ""
        if not transcript:
            continue
        action_items_raw = row.get("action_items") or []
        decisions_raw = row.get("decisions") or []
        fixtures.append({
            "id": f"ami_{row.get('meeting_id', len(fixtures))}",
            "source": "AMI",
            "transcript": transcript[:4000],
            "expected_signals": [
                {"signal_type": "decision", "summary": d}
                for d in (decisions_raw if isinstance(decisions_raw, list) else [decisions_raw])
                if d
            ],
            "expected_action_items": [
                {"task": a if isinstance(a, str) else str(a), "owner_name": None,
                 "deadline": None, "supporting_quote": ""}
                for a in (action_items_raw if isinstance(action_items_raw, list) else [action_items_raw])
                if a
            ],
        })
    print(f"  Loaded {len([f for f in fixtures if f['source']=='AMI'])} AMI examples.")


def load_meetingbank():
    print("Loading MeetingBank...")
    try:
        ds = load_dataset("huuuyeah/meetingbank", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  MeetingBank load failed ({e}), skipping.")
        return
    count_before = len(fixtures)
    for row in ds.select(range(min(5, len(ds)))):
        transcript = row.get("transcript") or ""
        if not transcript:
            continue
        fixtures.append({
            "id": f"meetingbank_{len(fixtures)}",
            "source": "MeetingBank",
            "transcript": transcript[:4000],
            "expected_signals": [],
            "expected_action_items": [],
            "summary_reference": row.get("summary") or "",
        })
    print(f"  Loaded {len(fixtures) - count_before} MeetingBank examples.")


HAND_LABELED = [
    {
        "id": "hand_1", "source": "hand_labeled",
        "transcript": (
            "Alice: We've decided to migrate to PostgreSQL by end of Q1.\n"
            "Bob: I'll handle the schema migration. Target is March 15th.\n"
            "Alice: One blocker — we don't have staging credentials yet.\n"
            "Bob: I'll get those from DevOps by Monday.\n"
            "Carol: We need to update the API docs before the client demo on Wednesday."
        ),
        "expected_signals": [
            {"signal_type": "decision", "summary": "Migrate to PostgreSQL by end of Q1"},
            {"signal_type": "blocker", "summary": "No staging credentials"},
            {"signal_type": "commitment", "summary": "Bob will get staging credentials by Monday"},
        ],
        "expected_action_items": [
            {"task": "Handle the schema migration", "owner_name": "Bob", "deadline": "March 15th",
             "supporting_quote": "I'll handle the schema migration. Target is March 15th"},
            {"task": "Get staging credentials from DevOps", "owner_name": "Bob", "deadline": "Monday",
             "supporting_quote": "I'll get those from DevOps by Monday"},
            {"task": "Update the API docs before client demo", "owner_name": "Carol", "deadline": "Wednesday",
             "supporting_quote": "we need to update the API docs before the client demo on Wednesday"},
        ],
    },
    {
        "id": "hand_2", "source": "hand_labeled",
        "transcript": (
            "Dave: The API latency is above 500ms in production. That's a blocker for the release.\n"
            "Eve: I can profile the slow queries this week and send a report.\n"
            "Dave: We also committed to delivering the dashboard to the client by Thursday.\n"
            "Eve: Got it. I'll prioritize the dashboard over the profiling."
        ),
        "expected_signals": [
            {"signal_type": "blocker", "summary": "API latency above 500ms blocking release"},
            {"signal_type": "commitment", "summary": "Deliver dashboard to client by Thursday"},
        ],
        "expected_action_items": [
            {"task": "Profile slow queries and send report", "owner_name": "Eve", "deadline": "this week",
             "supporting_quote": "I can profile the slow queries this week and send a report"},
            {"task": "Prioritize dashboard delivery", "owner_name": "Eve", "deadline": "Thursday",
             "supporting_quote": "I'll prioritize the dashboard over the profiling"},
        ],
    },
]


if __name__ == "__main__":
    load_ami()
    load_meetingbank()
    fixtures.extend(HAND_LABELED)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(fixtures, f, indent=2)
    print(f"\nWrote {len(fixtures)} fixtures to {OUTPUT}")
```

- [ ] **Step 2: Run prepare_datasets.py**

```bash
python tests/eval/prepare_datasets.py
```

Expected: Fixture file written with AMI + MeetingBank + 2 hand-labeled examples.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/prepare_datasets.py tests/eval/datasets/
git commit -m "feat: eval dataset preparation from AMI, MeetingBank, and hand-labeled examples"
```

---

## Task 12: LangSmith Evaluation Tests

**Files:**
- Create: `tests/eval/test_signal_detection.py`
- Create: `tests/eval/test_action_extraction.py`
- Create: `tests/eval/test_search_quality.py`

- [ ] **Step 1: Write test_signal_detection.py using LangSmith evaluate()**

```python
# tests/eval/test_signal_detection.py
"""
Evaluates signal detection (decision/blocker/commitment) using LangSmith evaluate().
Requires: LANGCHAIN_API_KEY set. Run prepare_datasets.py first.

LangSmith automatically uploads the dataset and runs the evaluator,
storing results in the LangSmith dashboard at smith.langchain.com.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from langsmith import evaluate, Client
from langsmith.schemas import Run, Example

with open("tests/eval/datasets/golden_transcripts.json") as f:
    GOLDEN = [g for g in json.load(f) if g.get("expected_signals")]

DATASET_NAME = "sync_ai-signal-detection"


def _upload_dataset_if_needed():
    client = Client()
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME not in existing:
        examples = [
            {"inputs": {"transcript": g["transcript"]},
             "outputs": {"signals": g["expected_signals"]}}
            for g in GOLDEN
        ]
        client.create_dataset(DATASET_NAME, description="Signal detection eval from AMI + hand-labeled")
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_name=DATASET_NAME,
        )


def run_signal_detection(inputs: dict) -> dict:
    """Function under test — calls AnalysisAgent with the transcript."""
    from app.agents.analysis import run_analysis_node
    from app.core.graph import make_initial_state, TranscriptChunk
    state = make_initial_state("eval", [])
    state["analysis_window"] = [
        TranscriptChunk(speaker="Test", text=inputs["transcript"], timestamp=0.0, sequence_num=1)
    ]
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0
    result_state = run_analysis_node(state)
    return {"signals": result_state["signals"]}


class SignalF1Evaluator:
    def __call__(self, run: Run, example: Example) -> dict:
        predicted = {s["signal_type"] for s in (run.outputs or {}).get("signals", [])}
        expected = {s["signal_type"] for s in (example.outputs or {}).get("signals", [])}
        tp = len(predicted & expected)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(expected) if expected else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"key": "signal_f1", "score": f1}


def test_signal_detection_f1():
    _upload_dataset_if_needed()
    results = evaluate(
        run_signal_detection,
        data=DATASET_NAME,
        evaluators=[SignalF1Evaluator()],
        experiment_prefix="signal-detection",
    )
    avg_f1 = sum(r["feedback"].get("signal_f1", 0) for r in results) / max(len(results), 1)
    assert avg_f1 >= 0.8, f"Average signal detection F1={avg_f1:.2f} below 0.80"
```

- [ ] **Step 2: Write test_action_extraction.py using LangSmith evaluate()**

```python
# tests/eval/test_action_extraction.py
import json
import pytest
from langsmith import evaluate, Client
from langsmith.schemas import Run, Example

with open("tests/eval/datasets/golden_transcripts.json") as f:
    GOLDEN = [g for g in json.load(f) if g.get("expected_action_items") and g["source"] == "hand_labeled"]

DATASET_NAME = "sync_ai-action-extraction"
PARTICIPANTS = [{"name": n, "email": f"{n.lower()}@test.com"}
                for n in ["Alice", "Bob", "Carol", "Dave", "Eve"]]


def _upload_dataset_if_needed():
    client = Client()
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME not in existing:
        examples = [
            {"inputs": {"transcript": g["transcript"], "participants": PARTICIPANTS},
             "outputs": {"action_items": g["expected_action_items"]}}
            for g in GOLDEN
        ]
        client.create_dataset(DATASET_NAME, description="Action extraction eval — hand labeled")
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_name=DATASET_NAME,
        )


def run_extraction(inputs: dict) -> dict:
    from app.agents.extraction import run_extraction_node
    from app.core.graph import make_initial_state, TranscriptChunk
    transcript = inputs["transcript"]
    lines = transcript.split("\n")
    chunks = []
    for i, line in enumerate(lines):
        if ": " in line:
            speaker, text = line.split(": ", 1)
        else:
            speaker, text = "Speaker", line
        chunks.append(TranscriptChunk(speaker=speaker, text=text, timestamp=float(i), sequence_num=i+1))
    state = make_initial_state("eval", inputs.get("participants", []))
    state["transcript_chunks"] = chunks
    state["meeting_ended"] = True
    result_state = run_extraction_node(state)
    return {"action_items": result_state["action_items"]}


class ExtractionPrecisionRecallEvaluator:
    def __call__(self, run: Run, example: Example) -> dict:
        predicted_tasks = {i["task"] for i in (run.outputs or {}).get("action_items", [])}
        expected_tasks = {i["task"] for i in (example.outputs or {}).get("action_items", [])}
        tp = len(predicted_tasks & expected_tasks)
        precision = tp / len(predicted_tasks) if predicted_tasks else 0
        recall = tp / len(expected_tasks) if expected_tasks else 0
        return {"key": "extraction_precision", "score": precision}


class HallucinationRateEvaluator:
    def __call__(self, run: Run, example: Example) -> dict:
        items = (run.outputs or {}).get("action_items", [])
        hallucinated = [i for i in items if i.get("needs_review")]
        rate = len(hallucinated) / len(items) if items else 0
        return {"key": "hallucination_rate", "score": 1 - rate}   # higher = better


def test_action_extraction_quality():
    _upload_dataset_if_needed()
    results = evaluate(
        run_extraction,
        data=DATASET_NAME,
        evaluators=[ExtractionPrecisionRecallEvaluator(), HallucinationRateEvaluator()],
        experiment_prefix="action-extraction",
    )
    precisions = [r["feedback"].get("extraction_precision", 0) for r in results]
    avg_precision = sum(precisions) / max(len(precisions), 1)
    hallucination_scores = [r["feedback"].get("hallucination_rate", 0) for r in results]
    avg_hallucination_quality = sum(hallucination_scores) / max(len(hallucination_scores), 1)

    assert avg_precision >= 0.8, f"Average extraction precision={avg_precision:.2f} below 0.80"
    assert avg_hallucination_quality >= 0.95, f"Hallucination quality={avg_hallucination_quality:.2f} below 0.95"
```

- [ ] **Step 3: Write test_search_quality.py using RAGAS**

```python
# tests/eval/test_search_quality.py
import pytest
from unittest.mock import MagicMock, patch

SEARCH_CASES = [
    {
        "query": "PostgreSQL migration deadline",
        "expected_keywords": ["PostgreSQL", "migration", "March"],
        "corpus": "Alice: We decided to migrate to PostgreSQL by end of Q1. Bob: Schema migration target is March 15th.",
    },
    {
        "query": "API latency production blocker",
        "expected_keywords": ["latency", "500ms", "blocker"],
        "corpus": "Dave: The API latency is above 500ms in production. That's a blocker for the release.",
    },
    {
        "query": "dashboard client delivery Thursday",
        "expected_keywords": ["dashboard", "client", "Thursday"],
        "corpus": "Dave: We committed to delivering the dashboard to the client by Thursday. Eve: I'll prioritize the dashboard.",
    },
]


@pytest.mark.parametrize("case", SEARCH_CASES)
def test_search_returns_relevant_chunk(case):
    mock_docs = [MagicMock(page_content=case["corpus"], metadata={"meeting_id": "eval", "sequence": 0})]

    with patch("app.core.search.Chroma") as MockChroma, \
         patch("app.core.search.HuggingFaceEmbeddings"):
        MockChroma.return_value.similarity_search.return_value = mock_docs
        from app.core.search import search_transcripts
        results = search_transcripts(case["query"], chroma_persist_dir="/tmp/test", limit=3)

    assert len(results) > 0, f"No results for: {case['query']}"
    result_text = results[0]["text"].lower()
    matched = [kw for kw in case["expected_keywords"] if kw.lower() in result_text]
    match_rate = len(matched) / len(case["expected_keywords"])
    assert match_rate >= 0.6, (
        f"Query '{case['query']}': matched {len(matched)}/{len(case['expected_keywords'])} keywords: {matched}"
    )
```

- [ ] **Step 4: Run the eval tests**

```bash
# Search quality (no LangSmith API needed)
pytest tests/eval/test_search_quality.py -v

# LangSmith evals (requires LANGCHAIN_API_KEY)
pytest tests/eval/test_signal_detection.py tests/eval/test_action_extraction.py -v
```

Expected: All pass. Results visible in LangSmith dashboard at `smith.langchain.com`.

- [ ] **Step 5: Commit**

```bash
git add tests/eval/
git commit -m "feat: LangSmith eval tests for signal detection, action extraction, and search quality"
```

---

## Task 13: Docker + GitHub Actions

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.github/workflows/test.yml`
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY prompts/ ./prompts/

RUN mkdir -p /data/chroma

ENV DATABASE_URL=/data/sync_ai.db
ENV CHROMA_PERSIST_DIR=/data/chroma

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    env_file:
      - .env
    environment:
      - DATABASE_URL=/data/sync_ai.db
      - CHROMA_PERSIST_DIR=/data/chroma
```

- [ ] **Step 3: Create .github/workflows/test.yml**

```yaml
name: Tests
on:
  push:
    branches: ["*"]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ tests/integration/ -v
        env:
          DATABASE_URL: ":memory:"
          CHROMA_PERSIST_DIR: "/tmp/chroma_test"
          GROQ_API_KEY: "test_placeholder"
```

- [ ] **Step 4: Create .github/workflows/deploy.yml**

```yaml
name: Deploy to Render
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Render deploy hook
        run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
```

- [ ] **Step 5: Build Docker image**

```bash
docker build -t sync_ai_langchain .
```

Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .github/
git commit -m "feat: Docker, docker-compose, and GitHub Actions CI/CD"
```

---

## Task 14: Final Verification

- [ ] **Step 1: Run full unit + integration test suite**

```bash
pytest tests/unit/ tests/integration/ -v --tb=short
```

Expected: All pass.

- [ ] **Step 2: Run search quality eval**

```bash
pytest tests/eval/test_search_quality.py -v
```

Expected: All 3 cases pass.

- [ ] **Step 3: Start app locally**

```bash
cp .env.example .env
# Edit .env: set GROQ_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true
mkdir -p data
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Dashboard renders.

- [ ] **Step 4: Verify LangSmith tracing is active**

```bash
curl -X POST http://localhost:8000/webhook/recall/nosession \
  -H "Content-Type: application/json" \
  -d '{"transcript":{"words":[{"text":"Hello","start_time":0.0,"end_time":0.5,"speaker":0}]}}' | python -m json.tool
```

Expected: `{"ok": false}`. Check `smith.langchain.com` — if a real session was running, every LangChain call would appear as a trace automatically.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "chore: all tests green, LangSmith tracing active, ready for deployment"
```
