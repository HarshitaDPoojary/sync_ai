# sync_ai

A meeting copilot that joins your calls, transcribes in real time, surfaces live suggestions, and automatically extracts and delivers action items to Slack and Gmail.

---

## What it does

- **Joins meetings as a bot** via Recall.ai — works with Zoom, Google Meet, and Teams without per-platform SDKs
- **Live dashboard** streams the transcript, follow-up questions, and detected decisions/blockers/commitments as the meeting happens
- **Extracts action items** from the full transcript with owner, deadline, and a verbatim supporting quote for verification
- **Delivers summaries** to a Slack channel and individual Gmail messages to assignees after the meeting ends
- **Semantic search** across all past meeting transcripts using local embeddings (no external vector API)
- **Multi-tenant** — each user connects their own Slack and Google accounts via OAuth; data is fully isolated per user

---

## Architecture

Five specialized agents communicate through an in-process event bus. No shared mutable state between agents.

```
Recall.ai webhook (POST /webhook/recall/{meeting_id})
        │
        ▼
┌─────────────────────┐
│   TranscriptAgent   │  Normalizes speaker labels, emits TranscriptChunkEvent
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    AnalysisAgent    │  5-minute sliding window → Groq LLM every 30s
└──────────┬──────────┘  Emits: SuggestionEvent, SignalEvent
           ▼
┌─────────────────────┐
│  ExtractionAgent    │  Full transcript at checkpoints + meeting end
└──────────┬──────────┘  Emits: ActionItemEvent, SummaryEvent
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐  ┌───────────────┐
│ Delivery │  │ StorageAgent  │
│  Agent   │  │               │
└──────────┘  └───────────────┘
 Slack/Gmail   ChromaDB + SQLite
```

Action items include a verbatim supporting quote from the transcript. A post-extraction step verifies each quote exists before the item is surfaced — items that fail are flagged `needs_review` rather than silently dropped.

---

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + WebSockets |
| Meeting bot | Recall.ai |
| LLM | Groq — `llama-3.1-70b-versatile` |
| Embeddings | `sentence-transformers` (local, no API cost) |
| Vector store | ChromaDB (embedded, persisted to disk) |
| Relational DB | SQLite via SQLModel |
| Auth | Clerk |
| Delivery | Slack SDK + Gmail API |
| Tracing | LangSmith |

---

## Quick start (Docker)

```bash
git clone <repo-url>
cd sync_ai
cp .env.example .env   # fill in required keys (see below)
docker compose up
```

Open `http://localhost:8000`. Sign in with Clerk, connect your Slack and Google accounts from the Settings page, then paste a meeting URL to start a session.

---

## Manual setup

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # fill in keys
python migrations/001_add_users.py

uvicorn app.main:app --reload
```

---

## Environment variables

Create a `.env` file at the repo root. All variables marked **required** must be set before the app will start.

### Recall.ai

| Variable | Required | Description |
|---|---|---|
| `RECALL_API_KEY` | yes | API key from the Recall.ai dashboard |
| `RECALL_WEBHOOK_SECRET` | yes | Webhook signing secret (`whsec_…`) — set after creating the webhook endpoint |
| `RECALL_BASE_URL` | no | Defaults to `https://us-west-2.recall.ai/api/v1` |
| `RECALL_TRANSCRIPTION_PROVIDER` | no | Transcription backend (`assembly_ai`, `deepgram`, etc.) |
| `RECALL_BOT_NAME` | no | Display name of the bot in the meeting |

### LLM

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes | API key from console.groq.com |
| `GROQ_MODEL` | no | Defaults to `llama-3.1-70b-versatile` |

### Embeddings

| Variable | Required | Description |
|---|---|---|
| `EMBEDDING_MODEL` | no | HuggingFace model name; defaults to `BAAI/bge-small-en-v1.5` |

### Auth (Clerk)

| Variable | Required | Description |
|---|---|---|
| `CLERK_PUBLISHABLE_KEY` | yes | Frontend publishable key (`pk_…`) |
| `CLERK_SECRET_KEY` | yes | Backend secret key (`sk_…`) |
| `CLERK_FRONTEND_API` | yes | Frontend API URL from Clerk dashboard |

### Slack OAuth

| Variable | Required | Description |
|---|---|---|
| `SLACK_CLIENT_ID` | yes | OAuth app client ID |
| `SLACK_CLIENT_SECRET` | yes | OAuth app client secret |
| `SLACK_OAUTH_REDIRECT_URI` | yes | Must match the redirect URI configured in your Slack app |

### Google OAuth (Gmail + Calendar)

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLIENT_ID` | yes | OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | yes | OAuth client secret |
| `GOOGLE_OAUTH_REDIRECT_URI` | yes | Must match the authorized redirect URI in Google Cloud Console |

### Storage

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | no | SQLite path; defaults to `./data/sync_ai.db` |
| `CHROMA_PERSIST_DIR` | no | ChromaDB persistence path; defaults to `./data/chroma` |

### Tracing (optional)

| Variable | Required | Description |
|---|---|---|
| `LANGCHAIN_API_KEY` | no | LangSmith API key — enables LLM call tracing |
| `LANGCHAIN_PROJECT` | no | LangSmith project name |
| `LANGCHAIN_TRACING_V2` | no | Set to `true` to enable tracing |

### Misc

| Variable | Required | Description |
|---|---|---|
| `WEBHOOK_BASE_URL` | yes | Public base URL of your app (used to register Recall.ai webhook) |
| `CALENDAR_POLL_INTERVAL_SECONDS` | no | How often to check Google Calendar for upcoming meetings; default `300` |
| `CALENDAR_BOT_DISPATCH_OFFSET_SECONDS` | no | How many seconds before a meeting starts to dispatch the bot; default `120` |

---

## Integrations setup

### Recall.ai

1. Create an account at [recall.ai](https://recall.ai) and grab your API key.
2. Deploy the app so it has a public URL (e.g. on Render).
3. In the Recall.ai dashboard, create a webhook pointing to `https://<your-domain>/webhook/recall/{meeting_id}`.
4. Copy the webhook signing secret into `RECALL_WEBHOOK_SECRET`.

### Clerk

1. Create an application at [clerk.com](https://clerk.com).
2. Copy the publishable key, secret key, and frontend API URL into your `.env`.
3. No additional configuration needed — user sign-up and sign-in are handled by Clerk's hosted components.

### Slack OAuth

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps).
2. Under "OAuth & Permissions", add the scopes: `chat:write`, `chat:write.public`, `channels:read`.
3. Add `<your-domain>/auth/slack/callback` as a redirect URI.
4. Copy the client ID and secret into `.env`.

Users connect their own Slack workspace from the Settings page — the bot posts to whatever channel they configure per meeting.

### Google (Gmail + Calendar)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com).
2. Enable the Gmail API and Google Calendar API.
3. Create an OAuth 2.0 client ID (Web application type).
4. Add `<your-domain>/auth/google/callback` as an authorized redirect URI.
5. Copy the client ID and secret into `.env`.

Users connect their Google account from the Settings page. The app requests `gmail.send` and `calendar.readonly` scopes.

---

## Running tests

```bash
pytest tests/unit/
pytest tests/integration/
```

The eval suite runs LLM-based quality checks against annotated golden transcripts and requires `GROQ_API_KEY` and `LANGCHAIN_API_KEY`:

```bash
pytest tests/eval/
```

---

## Deployment (Render.com)

1. Push to GitHub and connect the repo to a Render Web Service.
2. Set the build command to `docker build` and run command to `uvicorn app.main:app --host 0.0.0.0 --port 8000`, or just use the included `Dockerfile`.
3. Add a persistent disk mounted at `/app/data` (SQLite + ChromaDB live here).
4. Set all environment variables in the Render dashboard.
5. Update `WEBHOOK_BASE_URL` to your Render URL and re-register the Recall.ai webhook.

---

## Project structure

```
sync_ai/
├── app/
│   ├── main.py                  Routes, WebSocket handler, OAuth callbacks
│   ├── agents/
│   │   ├── analysis.py          AnalysisAgent — sliding window + LLM suggestions
│   │   ├── extraction.py        ExtractionAgent — action items + anti-hallucination
│   │   ├── delivery.py          DeliveryAgent — Slack + Gmail
│   │   └── storage.py           StorageAgent — ChromaDB + SQLite
│   ├── core/
│   │   ├── session.py           MeetingSession orchestrator
│   │   ├── graph.py             LangGraph pipeline definition
│   │   ├── recall.py            Recall.ai API client
│   │   ├── search.py            Semantic search
│   │   ├── calendar_poller.py   Google Calendar auto-join
│   │   └── config.py            Settings (pydantic-settings)
│   ├── auth/
│   │   └── clerk.py             Clerk JWT verification
│   ├── models/
│   │   └── db.py                SQLModel models
│   ├── repositories/            Data access layer
│   └── integrations/
│       ├── slack.py
│       ├── gmail.py
│       └── google_calendar.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/                    LLM quality evals (DeepEval + RAGAS)
├── migrations/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
