# sync_ai — Meeting Copilot: Full System Design

**Date:** 2026-04-28  
**Status:** Approved  

---

## Context

Meetings generate decisions, commitments, and action items that get lost in unstructured conversation. sync_ai transforms live meetings into structured, actionable workflows by combining real-time audio transcription, multi-agent AI analysis, and automated task delivery. The system joins meetings as a bot participant, continuously analyzes the conversation, surfaces live suggestions and signals, extracts action items, and routes them to the right people via Slack and Gmail — all without manual note-taking.

---

## 1. System Overview

sync_ai is a FastAPI web application with five specialized AI agents communicating through an event bus. A user starts a meeting session from a web dashboard by pasting a meeting URL. The system deploys a Recall.ai bot that joins the call, streams transcript chunks back to the app in real time, and runs them through the agent pipeline. The dashboard shows live suggestions and signals while the meeting is happening. After the meeting, summaries and action items are sent to Slack/Gmail automatically, and the transcript is stored for semantic search.

### Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Web framework | FastAPI + WebSockets | Async-native, lightweight, built-in WebSocket support |
| Meeting bot | Recall.ai | Unified API for Zoom/Meet/Teams; no per-platform SDK needed |
| Transcription | Recall.ai real-time transcript stream | Recall.ai transcribes audio server-side and delivers chunks via webhook; no local STT needed |
| LLM | Groq API — `llama-3.1-70b-versatile` | Free tier, fast inference, open-weight model |
| Embeddings | sentence-transformers `BAAI/bge-small-en-v1.5` | Fully local, no API cost, strong retrieval quality |
| Vector store | ChromaDB (embedded) | Open-source, persisted to disk, no separate server |
| Relational DB | SQLite via SQLModel | Zero config, embedded, sufficient for this scale |
| Event bus | asyncio.Queue (Redis-swappable) | In-process for simplicity, interface allows Redis swap |
| Delivery | Slack SDK + Gmail API | Standard libraries, free to use |
| Eval (offline) | DeepEval + RAGAS | Pytest-integrated LLM eval; RAG pipeline evaluation |
| Eval (online) | W&B Weave | Free tier, traces every LLM call in production |
| Hosting | Render.com | Free web service tier, Docker-native, persistent disk, WebSocket support |
| CI/CD | GitHub Actions | Free for public repos; runs tests and triggers Render deploy |
| Containerization | Docker | Single Dockerfile, consistent local/prod environment |

---

## 2. Architecture: Event-Driven Multi-Agent Pipeline

### Design Principle

Each agent is an isolated async service. Agents communicate only through typed Pydantic events on the event bus — no shared mutable state. This makes agents independently testable, independently scalable, and replaceable without touching other agents.

### Pipeline

```
Recall.ai webhook (POST /webhook/recall)
        │
        ▼
┌─────────────────────┐
│   TranscriptAgent   │  Receives raw chunks, validates, normalizes speaker labels
└──────────┬──────────┘  Emits: TranscriptChunkEvent
           │
           ▼
┌─────────────────────┐
│    AnalysisAgent    │  Maintains a 5-minute sliding window of transcript
└──────────┬──────────┘  Calls Groq LLM every 30s or on significant chunk arrival
           │             Emits: SuggestionEvent, SignalEvent
           ▼
┌─────────────────────┐
│  ExtractionAgent    │  Processes full transcript at checkpoints and meeting end
└──────────┬──────────┘  Extracts action items with anti-hallucination verification
           │             Emits: ActionItemEvent, SummaryEvent
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐  ┌───────────────┐
│ Delivery │  │ StorageAgent  │
│  Agent   │  │               │
└──────────┘  └───────────────┘
 Slack/Gmail   ChromaDB + SQLite
```

### EventBus

```python
# core/event_bus.py
class EventBus:
    subscribe(event_type: str, handler: Callable) -> None
    publish(event: BaseEvent) -> None
    # Internals: asyncio.Queue per event type
    # Interface stable: swapping to Redis requires only changing EventBus internals
```

### Agent Contract

Every agent implements:
```python
class BaseAgent:
    async def start(self) -> None   # subscribe to events, begin processing loop
    async def stop(self) -> None    # clean shutdown, flush buffers
```

### MeetingSession

`core/session.py` — orchestrates agent lifecycle for a single meeting:
- Instantiates all 5 agents with shared event bus
- Calls `start()` on all agents
- Holds reference to Recall.ai bot ID
- Calls `stop()` on all agents when `MeetingEndedEvent` fires

---

## 3. Agent Specifications

### TranscriptAgent (`agents/transcript.py`)
- **Input:** Recall.ai webhook POST payloads (raw transcript chunks)
- **Behavior:** Parses chunk JSON, normalizes speaker display names against participant list, appends to session transcript buffer, emits `TranscriptChunkEvent`
- **Output:** `TranscriptChunkEvent(meeting_id, speaker, text, timestamp, sequence_num)`

### AnalysisAgent (`agents/analysis.py`)
- **Input:** `TranscriptChunkEvent` stream
- **Behavior:**
  - Maintains a sliding window of the last 5 minutes of transcript
  - Triggers LLM call every 30 seconds or when the window accumulates 500+ new tokens
  - Single Groq call with a structured prompt asking for: 3 follow-up questions, 2 clarification points, 2 talking points, any decisions/blockers/commitments in the window
  - Returns structured JSON; parsed into typed events
- **Output:** `SuggestionEvent`, `SignalEvent`
- **Groq prompt pattern:** System message defines agent role + JSON schema; user message is the transcript window. Temperature = 0.3 for consistency.

### ExtractionAgent (`agents/extraction.py`)
- **Input:** `SignalEvent`, `TranscriptChunkEvent` (accumulates full transcript)
- **Behavior:**
  - Runs at 10-minute checkpoints and on `MeetingEndedEvent`
  - Calls Groq with full transcript to extract action items: task description, owner name, deadline, confidence
  - **Anti-hallucination:** Prompt requires a verbatim supporting quote per item. Post-processing verifies quote exists in transcript (case-insensitive substring match). Items failing verification are flagged as `needs_review`.
  - Resolves owner names to Recall.ai participant records (display name → email mapping)
- **Output:** `ActionItemEvent`, `SummaryEvent`

### DeliveryAgent (`agents/delivery.py`)
- **Input:** `ActionItemEvent`, `SummaryEvent`, explicit `/sessions/{id}/deliver` API call
- **Behavior:**
  - Holds action items in memory until delivery is triggered (auto on meeting end, or manual)
  - Sends formatted Slack message to configured channel with action items grouped by owner
  - Sends Gmail to each owner's email with their assigned tasks and meeting context
  - Records delivery status to SQLite
- **Output:** Slack message, Gmail; persists `ActionItem.sent_via` in DB

### StorageAgent (`agents/storage.py`)
- **Input:** All event types
- **Behavior:**
  - Writes `TranscriptChunkEvent` to SQLite incrementally
  - On `MeetingEndedEvent`: assembles full transcript, chunks into 200-word segments with 50-word overlap, embeds each chunk with `bge-small-en-v1.5`, upserts into ChromaDB `transcripts` collection
  - Embeds `ActionItemEvent` items into ChromaDB `action_items` collection
  - Persists `MeetingSummary` to SQLite
- **Output:** ChromaDB documents, SQLite rows

---

## 4. Data Models

### SQLite (SQLModel)

```python
Meeting(id: str, title: str, platform_url: str, status: str,
        started_at: datetime, ended_at: Optional[datetime], recall_bot_id: str)

Participant(id: str, meeting_id: str, display_name: str,
            email: Optional[str], slack_id: Optional[str])

Transcript(id: str, meeting_id: str, full_text: str, chunks_json: str)

ActionItem(id: str, meeting_id: str, task: str, owner_id: Optional[str],
           deadline: Optional[str], confidence: float, status: str,
           needs_review: bool, supporting_quote: str, sent_via: Optional[str])

MeetingSummary(id: str, meeting_id: str, decisions: List[str],
               blockers: List[str], commitments: List[str], next_steps: List[str])
```

### Pydantic Events (event_bus.py)

```python
TranscriptChunkEvent(meeting_id, speaker, text, timestamp, sequence_num)
SuggestionEvent(meeting_id, type: Literal["question","clarification","talking_point"], text, confidence)
SignalEvent(meeting_id, signal_type: Literal["decision","blocker","commitment"], summary, speaker, timestamp)
ActionItemEvent(meeting_id, task, owner_name, owner_email, owner_slack, deadline, confidence, supporting_quote, needs_review)
SummaryEvent(meeting_id, decisions, blockers, commitments, next_steps)
MeetingEndedEvent(meeting_id, duration_seconds)
```

### ChromaDB Collections

| Collection | Document | Metadata |
|---|---|---|
| `transcripts` | 200-word transcript chunk | meeting_id, timestamp, speaker, sequence |
| `action_items` | Action item task text | meeting_id, owner_name, deadline, confidence |

---

## 5. API Design

### REST Endpoints

```
POST   /sessions                     Start session; body: {meeting_url, title, participant_emails[]}
GET    /sessions/{id}                Get session status, summary, action items
DELETE /sessions/{id}               End session, remove Recall.ai bot
GET    /sessions/{id}/suggestions   Latest suggestions (last 10)
GET    /sessions/{id}/signals       All detected signals (decisions/blockers/commitments)
GET    /sessions/{id}/action-items  All extracted action items
POST   /sessions/{id}/deliver       Trigger Slack/Gmail delivery
POST   /sessions/{id}/feedback      Submit thumbs up/down on a suggestion or action item
GET    /search?q={query}&limit={n}  Semantic search across past transcripts
POST   /webhook/recall              Recall.ai transcript webhook receiver (internal)
```

### WebSocket

```
WS /ws/{session_id}
```
Server pushes JSON messages of shape `{type: "transcript"|"suggestion"|"signal"|"action_item", data: {...}}` as events fire. Dashboard JS connects on session start and renders updates in real time.

---

## 6. Web Dashboard

Single HTML file served by FastAPI (`app/static/index.html`). No build step, no framework — vanilla JS + CSS.

**Layout:**
- **Top bar:** Session title, status indicator, "End Meeting" button
- **Left panel (60%):** Live transcript — scrolling, speaker-labeled, auto-scrolls to bottom
- **Right panel (40%):**
  - "Suggestions" tab — follow-up questions, clarifications, talking points as cards with thumbs up/down
  - "Signals" tab — decision/blocker/commitment chips, timestamped
  - "Action Items" tab — task list with owner, deadline, needs_review badge, "Send" button
- **Bottom bar:** Search input for past meetings

---

## 7. Evaluation

### Offline Evaluation (`tests/eval/`)

**DeepEval** (pytest-integrated):
- `test_hallucination.py` — verifies ExtractionAgent output faithfulness: extracted items must be grounded in transcript
- `test_answer_relevance.py` — verifies AnalysisAgent suggestions are relevant to the transcript window
- `test_signal_detection.py` — F1 score against annotated golden transcripts for decision/blocker/commitment detection
- `test_action_extraction.py` — precision/recall of extracted action items vs. human-labeled ground truth

**RAGAS:**
- `test_search_quality.py` — evaluates semantic search (NDCG, context recall) on curated query-result pairs
- Faithfulness score: action item supporting quotes verified against transcript

**Golden transcript corpus** (`tests/eval/fixtures/`):
- 5 annotated test transcripts with labeled signals, action items, and expected suggestions
- Loaded in pytest fixtures; all eval tests run against these

### Online Evaluation (Production)

**W&B Weave tracing:**
- Every Groq LLM call is wrapped with `weave.op()` — logs inputs, outputs, latency, token count
- Dashboard in W&B shows per-session LLM quality metrics and latency trends

**User feedback:**
- `POST /sessions/{id}/feedback` records thumbs up/down per suggestion/action item
- Confidence calibration: tracked in SQLite; periodically reviewed to tune confidence thresholds

**Hallucination monitoring:**
- `needs_review=True` rate tracked per session; alerted if > 20% of action items flagged

---

## 8. Deployment

### Environment Variables

```
RECALL_API_KEY          Recall.ai API key
GROQ_API_KEY            Groq API key
SLACK_BOT_TOKEN         Slack bot OAuth token
SLACK_CHANNEL_ID        Default Slack channel for summaries
GMAIL_CREDENTIALS_JSON  Gmail OAuth2 credentials (base64-encoded JSON)
DATABASE_URL            SQLite path (default: ./data/sync_ai.db)
CHROMA_PERSIST_DIR      ChromaDB persistence path (default: ./data/chroma)
WANDB_API_KEY           Weights & Biases API key (optional, enables Weave tracing)
```

### Docker

Single-stage `Dockerfile` — Python 3.11-slim, installs all Python deps, exposes port 8000. `docker-compose.yml` for local development with volume mounts for persistent data. No local GPU or audio processing required — transcription is handled by Recall.ai.

### GitHub Actions

- `.github/workflows/test.yml` — runs `pytest tests/` on every push and PR
- `.github/workflows/deploy.yml` — on push to `main`, triggers Render.com deploy hook via `curl`

### Render.com

- Web Service running Docker container
- Persistent disk mounted at `/data` (free 1GB) for SQLite + ChromaDB
- Environment variables set in Render dashboard
- Recall.ai webhook URL: `https://<app>.onrender.com/webhook/recall`

---

## 9. Project Structure

```
sync_ai/
├── app/
│   ├── main.py                  FastAPI app, all routes, WebSocket handler
│   ├── agents/
│   │   ├── base.py              BaseAgent abstract class
│   │   ├── transcript.py        TranscriptAgent
│   │   ├── analysis.py          AnalysisAgent (Groq LLM, sliding window)
│   │   ├── extraction.py        ExtractionAgent (Groq LLM, anti-hallucination)
│   │   ├── delivery.py          DeliveryAgent (Slack + Gmail)
│   │   └── storage.py           StorageAgent (ChromaDB + SQLite)
│   ├── core/
│   │   ├── event_bus.py         EventBus class + all Event Pydantic models
│   │   ├── session.py           MeetingSession orchestrator
│   │   └── recall.py            Recall.ai API client wrapper
│   ├── models/
│   │   └── db.py                SQLModel database models + engine setup
│   ├── integrations/
│   │   ├── slack.py             Slack SDK wrapper
│   │   └── gmail.py             Gmail API wrapper
│   └── static/
│       └── index.html           Web dashboard (vanilla JS)
├── tests/
│   ├── unit/                    Unit tests per agent (mocked event bus)
│   ├── integration/             Integration tests (real SQLite, mock Recall.ai)
│   └── eval/
│       ├── fixtures/            Golden transcripts + annotated ground truth
│       ├── test_hallucination.py
│       ├── test_signal_detection.py
│       ├── test_action_extraction.py
│       └── test_search_quality.py
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

## 10. Verification Plan

1. **Unit tests:** Each agent tested in isolation with a mock event bus. Inject synthetic `TranscriptChunkEvent`s, assert correct events emitted.
2. **Integration test:** Replay a pre-recorded Recall.ai webhook payload (saved JSON fixture), run the full agent pipeline, verify SQLite rows and ChromaDB documents written correctly.
3. **Eval suite:** Run `pytest tests/eval/` against golden transcripts; assert hallucination rate < 5%, signal detection F1 > 0.80, search NDCG > 0.75.
4. **End-to-end:** Deploy to Render, start a test session with a Zoom meeting URL, verify Recall.ai bot joins, transcript appears in dashboard, action items extracted, Slack message sent.
5. **Search:** After a stored meeting, query `GET /search?q=...` with terms from the transcript; verify semantically relevant chunks returned.
