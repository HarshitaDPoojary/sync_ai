# sync_ai — Meeting Copilot: LangChain + LangSmith Design

**Date:** 2026-04-28  
**Status:** Alternative design (LangChain/LangSmith variant)  
**Compare with:** `2026-04-28-meeting-copilot-design.md` (event-driven baseline)

---

## Context

Same system as the baseline design — a live meeting copilot that joins calls via Recall.ai, transcribes in real time, generates suggestions, extracts action items, and routes them via Slack/Gmail. This variant uses LangChain as the agent framework and LangSmith for tracing, evaluation, and observability, replacing the custom event bus and W&B Weave.

**When to prefer this design over the baseline:**
- You want built-in LLM tracing without instrumenting every call manually
- You want LangSmith's eval dataset and prompt playground tooling
- You're comfortable with LangChain abstractions and their versioning churn

**When to prefer the baseline:**
- You want full control over agent behavior without framework leakage
- You want zero vendor lock-in on the orchestration layer
- You're evaluating this system as a portfolio/interview piece where custom architecture shows more

---

## 1. System Overview

Same external behavior as the baseline. The difference is internal: instead of a custom `EventBus` + `BaseAgent` pattern, the pipeline is implemented as a **LangChain LCEL (LangChain Expression Language) chain** for each processing step, orchestrated by **LangGraph** for stateful session management. LangSmith replaces W&B Weave for tracing and evaluation.

### Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Web framework | FastAPI + WebSockets | Same as baseline |
| Meeting bot | Recall.ai | Same as baseline |
| LLM | Groq API via `langchain-groq` | LangChain's Groq integration; `llama-3.1-70b-versatile` |
| Agent orchestration | LangGraph | Stateful graph for meeting session; each node is an agent step |
| Chain construction | LangChain LCEL | Composable prompt → LLM → output parser pipelines |
| Embeddings | `langchain-community` HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) | LangChain wrapper around sentence-transformers |
| Vector store | ChromaDB via `langchain-chroma` | Same store, accessed through LangChain's VectorStore interface |
| Relational DB | SQLite via SQLModel | Same as baseline |
| Delivery | Slack SDK + Gmail API | Same as baseline |
| Tracing + eval | LangSmith | Automatic tracing of all LangChain calls; eval datasets and online evals |
| Hosting | Render.com | Same as baseline |
| CI/CD | GitHub Actions | Same as baseline |
| Containerization | Docker | Same as baseline |

---

## 2. Architecture: LangGraph Session Graph

### Design Principle

A `MeetingSession` is modeled as a **LangGraph `StateGraph`** where state accumulates as the meeting progresses. Each node in the graph is a LangChain chain that reads from and writes to a shared `MeetingState` typed dict. Nodes are triggered by new transcript chunks arriving via the Recall.ai webhook.

### Session State

```python
class MeetingState(TypedDict):
    meeting_id: str
    participants: List[Participant]
    transcript_chunks: List[TranscriptChunk]    # full accumulated transcript
    analysis_window: List[TranscriptChunk]      # last 5 minutes
    suggestions: List[Suggestion]
    signals: List[Signal]                        # decisions/blockers/commitments
    action_items: List[ActionItem]
    summary: Optional[MeetingSummary]
    delivery_status: Optional[DeliveryStatus]
```

### Graph Structure

```
Recall.ai webhook → ingest_chunk node
                          │
                          ▼
                   should_analyze? ──── no ──── END (await next chunk)
                          │
                         yes
                          │
                          ▼
                   analysis_node  (AnalysisChain)
                          │
                          ▼
                   should_extract? ──── no ──── END
                          │
                         yes
                          │
                          ▼
                  extraction_node  (ExtractionChain)
                          │
                   meeting_ended? ──── no ──── END
                          │
                         yes
                          │
                     ┌────┴────┐
                     ▼         ▼
               delivery_node  storage_node
```

Each node is a pure function `(MeetingState) -> MeetingState` — LangGraph manages state threading automatically.

---

## 3. LangChain Chain Specifications

All chains use **LCEL** (`prompt | llm | output_parser`) with `langchain-groq` as the LLM provider.

### AnalysisChain (`agents/analysis.py`)

```python
analysis_chain = (
    ChatPromptTemplate.from_messages([
        ("system", ANALYSIS_SYSTEM_PROMPT),   # defines role + JSON schema
        ("human", "{transcript_window}")
    ])
    | ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3)
    | JsonOutputParser()                       # parses into SuggestionsAndSignals
)
```

- Input: last 5 minutes of transcript as formatted string
- Output: `{"suggestions": [...], "signals": [...]}`
- Triggered every 30s or on 500+ new tokens in window

### ExtractionChain (`agents/extraction.py`)

```python
extraction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),  # requires verbatim quotes
        ("human", "{full_transcript}")
    ])
    | ChatGroq(model="llama-3.1-70b-versatile", temperature=0.1)
    | JsonOutputParser()
)
```

- Input: full transcript
- Output: `{"action_items": [...], "summary": {...}}`
- Anti-hallucination: same post-processing substring verification as baseline
- Runs at 10-minute checkpoints and on meeting end

### SummaryChain (`agents/extraction.py`)

Separate chain focused on structured summary (decisions, blockers, commitments, next steps). Called once on `MeetingEndedEvent`.

```python
summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", "{full_transcript}")
    ])
    | ChatGroq(model="llama-3.1-70b-versatile", temperature=0.1)
    | PydanticOutputParser(pydantic_object=MeetingSummary)
)
```

### SearchChain (`core/search.py`)

Uses LangChain's `RetrievalQA` pattern for semantic search over past meetings:

```python
retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 5})
search_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(model="llama-3.1-70b-versatile"),
    retriever=retriever,
    return_source_documents=True
)
```

---

## 4. LangSmith Integration

### Automatic Tracing

Every LangChain/LangGraph call is automatically traced to LangSmith when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set. No manual instrumentation needed. Each trace captures:
- Full prompt sent to Groq
- Raw LLM response
- Latency, token count, cost estimate
- Parent chain context (which graph node triggered this call)

### Evaluation Datasets

LangSmith stores evaluation datasets as collections of (input, expected output) pairs. We create three datasets:

| Dataset | Input | Expected output | Metric |
|---|---|---|---|
| `signal-detection` | Transcript window | Labeled signals | F1 score |
| `action-extraction` | Full transcript | Labeled action items + quotes | Precision/Recall, Faithfulness |
| `search-quality` | Natural language query | Relevant transcript chunks | NDCG |

### Running Evals

```bash
# Run offline evals against LangSmith datasets
langchain eval run --dataset signal-detection --llm groq/llama-3.1-70b-versatile

# Or via pytest with LangSmith evaluator
pytest tests/eval/ --langsmith
```

### Online Monitoring

LangSmith's **online evaluation** runs lightweight evaluators on production traces automatically:
- `hallucination_evaluator` — checks if action items are grounded in the transcript (uses a small LLM judge)
- `relevance_evaluator` — checks if suggestions are relevant to the transcript window
- Alerts configured in LangSmith dashboard when evaluator scores drop below threshold

### Prompt Playground

LangSmith's playground lets you iterate on `ANALYSIS_SYSTEM_PROMPT` and `EXTRACTION_SYSTEM_PROMPT` interactively against real production traces without redeploying. This is the main UX advantage of LangSmith over W&B Weave.

---

## 5. Data Models

Same SQLite models as baseline. LangGraph session state is held in-memory per `MeetingSession` instance and flushed to SQLite on `MeetingEndedEvent`.

LangChain's `ConversationBufferWindowMemory` is **not used** — the `MeetingState.transcript_chunks` list serves as the authoritative transcript buffer, keeping state management explicit and testable.

---

## 6. API Design

Identical to baseline. The LangGraph graph is an internal implementation detail — the REST API and WebSocket interface are unchanged.

The one addition: a LangSmith trace URL is stored per session in SQLite for debugging:

```
GET /sessions/{id}/trace   Returns the LangSmith trace URL for this session
```

---

## 7. Web Dashboard

Identical to baseline — single `index.html`, vanilla JS, WebSocket for live updates.

---

## 8. Evaluation

### Offline (`tests/eval/`)

Uses LangSmith evaluators instead of DeepEval:

```python
# tests/eval/test_signal_detection.py
from langsmith import evaluate

results = evaluate(
    run_extraction_chain,          # function under test
    data="signal-detection",       # LangSmith dataset name
    evaluators=[SignalF1Evaluator],
    experiment_prefix="signal-detection"
)
assert results.summary["f1"] > 0.80
```

RAGAS is still used for search quality evaluation (it's provider-agnostic and excellent for RAG evals).

### Online (Production)

LangSmith automatic tracing + online evaluators replace W&B Weave. Same user feedback endpoint (`POST /sessions/{id}/feedback`) — scores stored in SQLite and also pushed to LangSmith as human feedback annotations.

---

## 9. Deployment

### Additional Environment Variables (vs. baseline)

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY            LangSmith API key (free tier available)
LANGCHAIN_PROJECT=sync_ai    Project name in LangSmith dashboard
```

All other env vars identical to baseline (RECALL_API_KEY, GROQ_API_KEY, etc.).

### Docker / Render / GitHub Actions

Identical to baseline. `langchain`, `langchain-groq`, `langchain-chroma`, `langchain-community`, and `langgraph` added to `requirements.txt`.

---

## 10. Project Structure

```
sync_ai/
├── app/
│   ├── main.py                  FastAPI app, routes, WebSocket handler
│   ├── agents/
│   │   ├── analysis.py          AnalysisChain (LCEL) + analysis graph node
│   │   ├── extraction.py        ExtractionChain + SummaryChain + extraction node
│   │   ├── delivery.py          Delivery node (Slack + Gmail, same as baseline)
│   │   └── storage.py           Storage node (ChromaDB + SQLite, same as baseline)
│   ├── core/
│   │   ├── graph.py             LangGraph StateGraph definition + MeetingState
│   │   ├── session.py           MeetingSession: instantiates + runs graph per meeting
│   │   ├── recall.py            Recall.ai API client wrapper (same as baseline)
│   │   └── search.py            RetrievalQA chain for semantic search
│   ├── models/
│   │   └── db.py                SQLModel database models (same as baseline)
│   ├── integrations/
│   │   ├── slack.py             Slack SDK wrapper (same as baseline)
│   │   └── gmail.py             Gmail API wrapper (same as baseline)
│   └── static/
│       └── index.html           Web dashboard (same as baseline)
├── tests/
│   ├── unit/                    Unit tests per chain (mocked ChatGroq)
│   ├── integration/             Integration tests (real SQLite, mock Recall.ai webhook)
│   └── eval/
│       ├── datasets/            LangSmith dataset fixtures (JSON)
│       ├── test_signal_detection.py   LangSmith evaluator
│       ├── test_action_extraction.py  LangSmith evaluator
│       └── test_search_quality.py     RAGAS evaluator
├── prompts/
│   ├── analysis_system.txt      AnalysisChain system prompt (version-controlled)
│   ├── extraction_system.txt    ExtractionChain system prompt
│   └── summary_system.txt       SummaryChain system prompt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        ├── test.yml
        └── deploy.yml
```

Note: `core/event_bus.py` and `agents/base.py` from the baseline are replaced by `core/graph.py` (LangGraph state graph). The `prompts/` directory is added — externalizing prompts makes them editable in LangSmith playground without touching Python files.

---

## 11. Trade-off Summary vs. Baseline

| Concern | Baseline (event-driven) | This design (LangChain/LangSmith) |
|---|---|---|
| **Tracing setup** | Manual W&B Weave instrumentation | Zero-config: set env vars, everything traced |
| **Prompt iteration** | Edit Python file, redeploy | LangSmith playground on live traces, no redeploy |
| **Eval tooling** | DeepEval (pytest-native) | LangSmith evaluators (cloud-native) |
| **Testability** | Excellent — pure event handlers | Good — mock ChatGroq, test LCEL chains |
| **Framework coupling** | None — pure Python | Coupled to LangChain; breaking changes are frequent |
| **Observability** | W&B Weave (good) | LangSmith (better for prompt-focused debugging) |
| **Complexity** | Medium — custom bus is ~100 lines | Medium — LangGraph adds state graph concepts |
| **Production stability** | High — no framework risk | Medium — LangChain 0.x → 1.x migrations have been painful |
| **Portfolio signal** | Shows systems design fundamentals | Shows LLM framework knowledge |

---

## 12. Verification Plan

1. **Unit tests:** Mock `ChatGroq` with `langchain_core.messages.AIMessage`. Test each LCEL chain with synthetic transcript inputs, assert output schema matches Pydantic models.
2. **Graph test:** Run the full `StateGraph` against a replayed Recall.ai webhook fixture; assert all state fields populated correctly after graph completion.
3. **LangSmith eval:** Run `pytest tests/eval/` — asserts F1 > 0.80 on signal detection, faithfulness > 0.95 on action extraction.
4. **RAGAS search eval:** Assert NDCG > 0.75 on curated query-result pairs.
5. **End-to-end:** Deploy to Render, start session, verify LangSmith dashboard shows traces for the session, action items extracted, Slack message sent.
6. **Prompt playground:** Open a production trace in LangSmith, tweak `extraction_system.txt` in the playground, verify improved output without code changes.
