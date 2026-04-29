import hashlib
import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("sync_ai")

load_dotenv()

from app.core.search import semantic_search
from app.core.session import MeetingSession
from app.models.db import create_db_and_tables
from app.repositories.action_item_repo import ActionItemRepository
from app.repositories.meeting_repo import MeetingRepository

_SESSION_NOT_FOUND = "Session not found"

# Active sessions: meeting_id -> MeetingSession
_sessions: Dict[str, MeetingSession] = {}
# WebSocket connections per session: meeting_id -> list of WebSocket
_ws_connections: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.makedirs("data", exist_ok=True)
    create_db_and_tables()
    # Warm up the embedding model so the first meeting doesn't stall 60s on model download
    from app.core.search import _get_vectorstore
    from app.core.config import get_settings
    _settings = get_settings()
    _get_vectorstore(_settings.embedding_model, _settings.chroma_persist_dir)
    logger.info("embedding_model_loaded model=%s", _settings.embedding_model)
    yield


app = FastAPI(title="sync_ai Meeting Copilot", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# ── Request schemas ────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    meeting_url: str
    title: str
    participant_emails: List[str] = []
    slack_channel_id: Optional[str] = None  # overrides SLACK_CHANNEL_ID env var


class FeedbackRequest(BaseModel):
    item_id: str
    item_type: str  # "suggestion" | "action_item"
    rating: int = Field(..., ge=-1, le=1)  # 1 = thumbs up, -1 = thumbs down


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Dashboard ──────────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (_STATIC_DIR / "index.html").read_text()


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def start_session(body: StartSessionRequest):
    meeting_id = str(uuid.uuid4())

    session = MeetingSession(
        meeting_id=meeting_id,
        meeting_url=body.meeting_url,
        title=body.title,
        participant_emails=body.participant_emails,
        slack_channel_id=body.slack_channel_id,
    )
    # Start bot first — only persist to DB if Recall.ai accepts it
    await session.start()

    repo = MeetingRepository()
    repo.create(
        meeting_id=meeting_id,
        title=body.title,
        platform_url=body.meeting_url,
        recall_bot_id=session.bot_id or "",
        slack_channel_id=body.slack_channel_id,
    )
    repo.add_participants(meeting_id, body.participant_emails)

    _sessions[meeting_id] = session
    _ws_connections[meeting_id] = []

    logger.info("session_started meeting_id=%s bot_id=%s", meeting_id, session.bot_id)
    return {"meeting_id": meeting_id, "bot_id": session.bot_id, "status": "active"}


@app.get("/sessions/{meeting_id}")
def get_session(meeting_id: str):
    meeting = MeetingRepository().get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    # In-memory state may be absent after a server restart; fall back gracefully
    session = _sessions.get(meeting_id)
    state = session.get_state() if session else {}
    return {
        "meeting_id": meeting_id,
        "title": meeting.title,
        "status": meeting.status,
        "suggestions": state.get("suggestions", []),
        "signals": state.get("signals", []),
        "action_items": state.get("action_items", []),
        "summary": state.get("summary"),
    }


@app.delete("/sessions/{meeting_id}")
async def end_session(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    await session.stop()
    del _sessions[meeting_id]

    MeetingRepository().mark_ended(meeting_id)
    logger.info("session_ended meeting_id=%s", meeting_id)

    await _broadcast(meeting_id, {"type": "meeting_ended", "data": {}})
    return {"status": "ended"}


@app.get("/sessions/{meeting_id}/suggestions")
def get_suggestions(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"suggestions": session.get_state()["suggestions"][-10:]}


@app.get("/sessions/{meeting_id}/signals")
def get_signals(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"signals": session.get_state()["signals"]}


@app.get("/sessions/{meeting_id}/action-items")
def get_action_items(meeting_id: str):
    session = _sessions.get(meeting_id)
    if session:
        return {"action_items": session.get_state()["action_items"]}
    # Session not in memory (e.g. server restarted) — read from DB
    if not MeetingRepository().get(meeting_id):
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"action_items": [dict(i) for i in ActionItemRepository().list_by_meeting(meeting_id)]}


@app.post("/sessions/{meeting_id}/deliver")
async def trigger_delivery(meeting_id: str):
    import asyncio
    from app.agents.delivery import run_delivery_node
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    state = session.get_state()
    title = session.title
    slack_channel_id = session._slack_channel_id
    asyncio.get_running_loop().run_in_executor(
        None,
        lambda: run_delivery_node(state, meeting_title=title, slack_channel_id=slack_channel_id),
    )
    return {"status": "delivery_queued"}


@app.post("/sessions/{meeting_id}/feedback")
def submit_feedback(meeting_id: str, body: FeedbackRequest):
    if body.item_type == "action_item":
        repo = ActionItemRepository()
        items = repo.list_by_meeting(meeting_id)
        if not any(str(i.id) == body.item_id for i in items):
            raise HTTPException(status_code=404, detail="Action item not found")
        status = "accepted" if body.rating > 0 else "rejected"
        repo.update_status(body.item_id, status)
    return {"status": "recorded"}


@app.get("/sessions/{meeting_id}/trace")
def get_trace_url(meeting_id: str):
    trace_url = MeetingRepository().get_trace_url(meeting_id)
    if trace_url is None and not MeetingRepository().get(meeting_id):
        raise HTTPException(status_code=404, detail="Not found")
    return {"trace_url": trace_url}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.get("/search")
def search(q: str, limit: int = Query(default=5, ge=1, le=50)):
    return {"results": semantic_search(q, limit=limit)}


# ── Recall.ai webhook ──────────────────────────────────────────────────────────

def _verify_recall_signature(request_body: bytes, signature_header: str, secret: str) -> bool:
    """Verify Recall.ai HMAC-SHA256 webhook signature."""
    expected = hmac.new(secret.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@app.post("/webhook/recall/{meeting_id}")
async def recall_webhook(meeting_id: str, request: Request, payload: Dict[str, Any]):
    from app.core.config import get_settings
    settings = get_settings()
    if settings.recall_webhook_secret:
        sig = request.headers.get("X-Recall-Signature", "")
        body = await request.body()
        if not _verify_recall_signature(body, sig, settings.recall_webhook_secret):
            logger.warning("webhook_invalid_signature meeting_id=%s", meeting_id)
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    session = _sessions.get(meeting_id)
    if not session:
        return {"ok": False}

    transcript = payload.get("transcript", {})
    words = transcript.get("words", [])
    if not words:
        return {"ok": True}

    speaker_id = words[0].get("speaker", 0)
    text = " ".join(w.get("text", "") for w in words if w.get("text"))
    timestamp = words[0].get("start_time", 0.0)
    speaker_label = f"Speaker {speaker_id}"

    import asyncio
    state_before = session.get_state()
    await asyncio.get_running_loop().run_in_executor(
        None, lambda: session.ingest_chunk(speaker=speaker_label, text=text, timestamp=timestamp)
    )
    state_after = session.get_state()

    await _broadcast(meeting_id, {
        "type": "transcript",
        "data": {"speaker": speaker_label, "text": text, "timestamp": timestamp},
    })

    for s in state_after["suggestions"][len(state_before["suggestions"]):]:
        await _broadcast(meeting_id, {"type": "suggestion", "data": dict(s)})

    for sig in state_after["signals"][len(state_before["signals"]):]:
        await _broadcast(meeting_id, {"type": "signal", "data": dict(sig)})

    for item in state_after["action_items"][len(state_before["action_items"]):]:
        await _broadcast(meeting_id, {"type": "action_item", "data": dict(item)})

    return {"ok": True}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    conns = _ws_connections.setdefault(meeting_id, [])
    if websocket not in conns:
        conns.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_connections.get(meeting_id, []):
            _ws_connections[meeting_id].remove(websocket)


async def _broadcast(meeting_id: str, message: Dict[str, Any]) -> None:
    conns = _ws_connections.get(meeting_id, [])
    dead = []
    for ws in conns:
        try:
            await ws.send_json(message)
        except Exception as exc:
            logger.debug("ws_send_failed meeting_id=%s error=%s", meeting_id, exc)
            dead.append(ws)
    for ws in dead:
        conns.remove(ws)
