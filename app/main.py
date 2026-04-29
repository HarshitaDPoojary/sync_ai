import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

load_dotenv()

from app.core.config import get_settings
from app.core.search import semantic_search
from app.core.session import MeetingSession
from app.models.db import ActionItem, Meeting, MeetingSummary, Participant, Transcript, create_db_and_tables, get_engine

# Active sessions: meeting_id -> MeetingSession
_sessions: Dict[str, MeetingSession] = {}
# WebSocket connections per session: meeting_id -> list of WebSocket
_ws_connections: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    create_db_and_tables()
    yield


app = FastAPI(title="sync_ai Meeting Copilot", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ── Request / Response schemas ─────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    meeting_url: str
    title: str
    participant_emails: List[str] = []


class FeedbackRequest(BaseModel):
    item_id: str
    item_type: str  # "suggestion" | "action_item"
    rating: int     # 1 = thumbs up, -1 = thumbs down


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("app/static/index.html") as f:
        return f.read()


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def start_session(body: StartSessionRequest):
    settings = get_settings()
    meeting_id = str(uuid.uuid4())

    session = MeetingSession(
        meeting_id=meeting_id,
        meeting_url=body.meeting_url,
        title=body.title,
        participant_emails=body.participant_emails,
    )
    await session.start()
    _sessions[meeting_id] = session
    _ws_connections[meeting_id] = []

    engine = get_engine()
    with Session(engine) as db:
        meeting = Meeting(
            id=meeting_id,
            title=body.title,
            platform_url=body.meeting_url,
            status="active",
            recall_bot_id=session.bot_id or "",
        )
        db.add(meeting)
        for email in body.participant_emails:
            db.add(Participant(
                id=str(uuid.uuid4()),
                meeting_id=meeting_id,
                display_name=email.split("@")[0],
                email=email,
            ))
        db.commit()

    return {"meeting_id": meeting_id, "bot_id": session.bot_id, "status": "active"}


@app.get("/sessions/{meeting_id}")
def get_session(meeting_id: str):
    engine = get_engine()
    with Session(engine) as db:
        meeting = db.get(Meeting, meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="Session not found")
    state = _sessions[meeting_id].get_state() if meeting_id in _sessions else {}
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
        raise HTTPException(status_code=404, detail="Session not found")
    await session.stop()
    del _sessions[meeting_id]

    engine = get_engine()
    with Session(engine) as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting:
            from datetime import datetime
            meeting.status = "ended"
            meeting.ended_at = datetime.utcnow()
            db.add(meeting)
            db.commit()

    await _broadcast(meeting_id, {"type": "meeting_ended", "data": {}})
    return {"status": "ended"}


@app.get("/sessions/{meeting_id}/suggestions")
def get_suggestions(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    suggestions = session.get_state()["suggestions"]
    return {"suggestions": suggestions[-10:]}


@app.get("/sessions/{meeting_id}/signals")
def get_signals(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"signals": session.get_state()["signals"]}


@app.get("/sessions/{meeting_id}/action-items")
def get_action_items(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"action_items": session.get_state()["action_items"]}


@app.post("/sessions/{meeting_id}/deliver")
def trigger_delivery(meeting_id: str):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from app.agents.delivery import run_delivery_node
    run_delivery_node(session.get_state(), meeting_title=session.title)
    return {"status": "delivered"}


@app.post("/sessions/{meeting_id}/feedback")
def submit_feedback(meeting_id: str, body: FeedbackRequest):
    engine = get_engine()
    with Session(engine) as db:
        if body.item_type == "action_item":
            item = db.get(ActionItem, body.item_id)
            if item:
                item.status = "accepted" if body.rating > 0 else "rejected"
                db.add(item)
                db.commit()
    return {"status": "recorded"}


@app.get("/sessions/{meeting_id}/trace")
def get_trace_url(meeting_id: str):
    engine = get_engine()
    with Session(engine) as db:
        meeting = db.get(Meeting, meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="Not found")
    return {"trace_url": meeting.langsmith_trace_url}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.get("/search")
def search(q: str, limit: int = 5):
    results = semantic_search(q, limit=limit)
    return {"results": results}


# ── Recall.ai webhook ──────────────────────────────────────────────────────────

@app.post("/webhook/recall/{meeting_id}")
async def recall_webhook(meeting_id: str, payload: Dict[str, Any]):
    session = _sessions.get(meeting_id)
    if not session:
        return {"ok": False}

    transcript = payload.get("transcript", {})
    words = transcript.get("words", [])
    if not words:
        return {"ok": True}

    speaker_id = words[0].get("speaker", 0)
    text = " ".join(w["text"] for w in words)
    timestamp = words[0].get("start_time", 0.0)
    speaker_label = f"Speaker {speaker_id}"

    state_before = session.get_state()
    session.ingest_chunk(speaker=speaker_label, text=text, timestamp=timestamp)
    state_after = session.get_state()

    chunk_event = {
        "type": "transcript",
        "data": {"speaker": speaker_label, "text": text, "timestamp": timestamp},
    }
    await _broadcast(meeting_id, chunk_event)

    new_suggestions = state_after["suggestions"][len(state_before["suggestions"]):]
    for s in new_suggestions:
        await _broadcast(meeting_id, {"type": "suggestion", "data": dict(s)})

    new_signals = state_after["signals"][len(state_before["signals"]):]
    for sig in new_signals:
        await _broadcast(meeting_id, {"type": "signal", "data": dict(sig)})

    new_items = state_after["action_items"][len(state_before["action_items"]):]
    for item in new_items:
        await _broadcast(meeting_id, {"type": "action_item", "data": dict(item)})

    return {"ok": True}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    _ws_connections.setdefault(meeting_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_connections[meeting_id].remove(websocket)


async def _broadcast(meeting_id: str, message: Dict[str, Any]) -> None:
    conns = _ws_connections.get(meeting_id, [])
    dead = []
    for ws in conns:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns.remove(ws)
