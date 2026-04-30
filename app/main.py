import asyncio
import base64
import hashlib
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt as pyjwt
from pydantic import BaseModel, Field

logger = logging.getLogger("sync_ai")

load_dotenv()

from app.auth.clerk import get_current_user
from app.core.config import get_settings, validate_settings
from app.core.search import semantic_search
from app.core.session import MeetingSession
from app.models.db import User, UserIntegration, create_db_and_tables
from app.repositories.action_item_repo import ActionItemRepository
from app.repositories.meeting_repo import MeetingRepository
from app.repositories.user_repo import UserRepository

_SESSION_NOT_FOUND = "Session not found"

# Active sessions: meeting_id -> MeetingSession
_sessions: Dict[str, MeetingSession] = {}
# WebSocket connections per session: meeting_id -> list of WebSocket
_ws_connections: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_settings(get_settings())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.makedirs("data", exist_ok=True)
    # Run migration (idempotent — safe to run on every startup)
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "migrations/001_add_users.py",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception:
        pass
    create_db_and_tables()
    # Start calendar auto-join poller
    from app.core.calendar_poller import poll_all_users_calendars
    _poll_task = asyncio.create_task(poll_all_users_calendars(_sessions))
    yield
    _poll_task.cancel()
    await asyncio.gather(_poll_task, return_exceptions=True)


app = FastAPI(title="sync_ai Meeting Copilot", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_STATIC_DIR = Path(__file__).parent / "static"


# ── Request schemas ────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    meeting_url: str
    title: str
    participant_emails: List[str] = []
    slack_channel_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    item_id: str
    item_type: str  # "suggestion" | "action_item"
    rating: int = Field(..., ge=-1, le=1)


# ── Public routes (no auth) ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_public_config():
    settings = get_settings()
    return {
        "clerk_publishable_key": settings.clerk_publishable_key,
        "clerk_frontend_api": settings.clerk_frontend_api,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/settings", response_class=HTMLResponse)
def settings_page():
    return (_STATIC_DIR / "settings.html").read_text()


@app.get("/meeting", response_class=HTMLResponse)
def meeting_page():
    return (_STATIC_DIR / "meeting.html").read_text()


# ── User ───────────────────────────────────────────────────────────────────────

@app.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "name": current_user.name}


@app.get("/integrations")
async def get_integrations(current_user: User = Depends(get_current_user)):
    repo = UserRepository()
    return {
        "slack": repo.get_integration(current_user.id, "slack") is not None,
        "gmail": repo.get_integration(current_user.id, "gmail") is not None,
        "google_calendar": repo.get_integration(current_user.id, "google_calendar") is not None,
    }


# ── Meetings list ──────────────────────────────────────────────────────────────

@app.get("/meetings")
def list_meetings(current_user: User = Depends(get_current_user)):
    meetings = MeetingRepository().list_by_user(current_user.id)
    return {"meetings": [
        {"id": m.id, "title": m.title, "status": m.status,
         "started_at": m.started_at.isoformat()}
        for m in meetings
    ]}


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def start_session(
    body: StartSessionRequest,
    current_user: User = Depends(get_current_user),
):
    meeting_id = str(uuid.uuid4())
    settings = get_settings()

    # Resolve per-user Slack token
    user_repo = UserRepository()
    slack_integration = user_repo.get_integration(current_user.id, "slack")
    slack_token = slack_integration.access_token if slack_integration else settings.slack_bot_token or None

    session = MeetingSession(
        meeting_id=meeting_id,
        meeting_url=body.meeting_url,
        title=body.title,
        participant_emails=body.participant_emails,
        slack_channel_id=body.slack_channel_id,
        slack_bot_token=slack_token,
        user_id=current_user.id,
    )
    await session.start()

    repo = MeetingRepository()
    repo.create(
        meeting_id=meeting_id,
        title=body.title,
        platform_url=body.meeting_url,
        recall_bot_id=session.bot_id or "",
        user_id=current_user.id,
        slack_channel_id=body.slack_channel_id,
    )
    repo.add_participants(meeting_id, body.participant_emails)

    _sessions[meeting_id] = session
    _ws_connections[meeting_id] = []

    logger.info("session_started meeting_id=%s user_id=%s bot_id=%s",
                meeting_id, current_user.id, session.bot_id)
    return {"meeting_id": meeting_id, "bot_id": session.bot_id, "status": "active"}


@app.get("/sessions/{meeting_id}")
def get_session(meeting_id: str, current_user: User = Depends(get_current_user)):
    meeting = MeetingRepository().get(meeting_id, user_id=current_user.id)
    if not meeting:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
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
async def end_session(meeting_id: str, current_user: User = Depends(get_current_user)):
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your session")
    await session.stop()
    del _sessions[meeting_id]

    MeetingRepository().mark_ended(meeting_id)
    logger.info("session_ended meeting_id=%s user_id=%s", meeting_id, current_user.id)

    await _broadcast(meeting_id, {"type": "meeting_ended", "data": {}})
    return {"status": "ended"}


@app.get("/sessions/{meeting_id}/suggestions")
def get_suggestions(meeting_id: str, current_user: User = Depends(get_current_user)):
    session = _sessions.get(meeting_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"suggestions": session.get_state()["suggestions"][-10:]}


@app.get("/sessions/{meeting_id}/signals")
def get_signals(meeting_id: str, current_user: User = Depends(get_current_user)):
    session = _sessions.get(meeting_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"signals": session.get_state()["signals"]}


@app.get("/sessions/{meeting_id}/action-items")
def get_action_items(meeting_id: str, current_user: User = Depends(get_current_user)):
    session = _sessions.get(meeting_id)
    if session:
        if session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
        return {"action_items": session.get_state()["action_items"]}
    if not MeetingRepository().get(meeting_id, user_id=current_user.id):
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return {"action_items": [dict(i) for i in ActionItemRepository().list_by_meeting(meeting_id)]}


@app.post("/sessions/{meeting_id}/deliver")
async def trigger_delivery(meeting_id: str, current_user: User = Depends(get_current_user)):
    from app.agents.delivery import run_delivery_node
    session = _sessions.get(meeting_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    state = session.get_state()
    title = session.title
    slack_channel_id = session._slack_channel_id
    slack_bot_token = session._slack_bot_token
    # Resolve per-user Gmail integration
    gmail_integration = UserRepository().get_integration(current_user.id, "gmail")
    asyncio.get_running_loop().run_in_executor(
        None,
        lambda: run_delivery_node(
            state,
            meeting_title=title,
            slack_channel_id=slack_channel_id,
            slack_bot_token=slack_bot_token,
            gmail_integration=gmail_integration,
        ),
    )
    return {"status": "delivery_queued"}


@app.post("/sessions/{meeting_id}/feedback")
def submit_feedback(
    meeting_id: str,
    body: FeedbackRequest,
    current_user: User = Depends(get_current_user),
):
    if not MeetingRepository().get(meeting_id, user_id=current_user.id):
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    if body.item_type == "action_item":
        repo = ActionItemRepository()
        items = repo.list_by_meeting(meeting_id)
        if not any(str(i.id) == body.item_id for i in items):
            raise HTTPException(status_code=404, detail="Action item not found")
        status = "accepted" if body.rating > 0 else "rejected"
        repo.update_status(body.item_id, status)
    return {"status": "recorded"}


@app.get("/sessions/{meeting_id}/trace")
def get_trace_url(meeting_id: str, current_user: User = Depends(get_current_user)):
    meeting = MeetingRepository().get(meeting_id, user_id=current_user.id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Not found")
    return {"trace_url": meeting.langsmith_trace_url}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.get("/search")
def search(
    q: str,
    limit: int = Query(default=5, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    return {"results": semantic_search(q, user_id=current_user.id, limit=limit)}


# ── Slack OAuth ────────────────────────────────────────────────────────────────

@app.get("/auth/slack")
async def slack_oauth_start(current_user: User = Depends(get_current_user)):
    settings = get_settings()
    state = pyjwt.encode(
        {"user_id": current_user.id, "exp": time.time() + 600},
        settings.clerk_secret_key, algorithm="HS256",
    )
    params = urlencode({
        "client_id": settings.slack_client_id,
        "scope": "chat:write,chat:write.public,channels:read",
        "redirect_uri": settings.slack_oauth_redirect_uri,
        "state": state,
    })
    url = f"https://slack.com/oauth/v2/authorize?{params}"
    return RedirectResponse(url)


@app.get("/auth/slack/callback")
async def slack_oauth_callback(code: str, state: str):
    settings = get_settings()
    try:
        claims = pyjwt.decode(state, settings.clerk_secret_key, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid state")
    user_id = claims["user_id"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://slack.com/api/oauth.v2.access",
            data={
                "code": code,
                "client_id": settings.slack_client_id,
                "client_secret": settings.slack_client_secret,
                "redirect_uri": settings.slack_oauth_redirect_uri,
            },
        )
    data = resp.json()
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Slack OAuth failed: {data.get('error')}")
    UserRepository().upsert_integration(
        user_id=user_id,
        provider="slack",
        access_token=data["access_token"],
        team_id=data["team"]["id"],
    )
    logger.info("slack_connected user_id=%s team_id=%s", user_id, data["team"]["id"])
    return RedirectResponse("/settings?connected=slack")


# ── Google OAuth (Gmail + Calendar) ───────────────────────────────────────────

@app.get("/auth/google")
async def google_oauth_start(current_user: User = Depends(get_current_user)):
    settings = get_settings()
    state = pyjwt.encode(
        {"user_id": current_user.id, "exp": time.time() + 600},
        settings.clerk_secret_key, algorithm="HS256",
    )
    scopes = " ".join([
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/calendar.readonly",
    ])
    params = urlencode({
        "client_id": settings.google_client_id,
        "response_type": "code",
        "redirect_uri": settings.google_oauth_redirect_uri,
        "scope": scopes,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    })
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
    return RedirectResponse(url)


@app.get("/auth/google/callback")
async def google_oauth_callback(code: str, state: str):
    settings = get_settings()
    try:
        claims = pyjwt.decode(state, settings.clerk_secret_key, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid state")
    user_id = claims["user_id"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=400, detail=f"Google OAuth failed: {data['error']}")
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))
    repo = UserRepository()
    for provider in ["gmail", "google_calendar"]:
        repo.upsert_integration(
            user_id=user_id,
            provider=provider,
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_expires_at=expires_at,
        )
    logger.info("google_connected user_id=%s", user_id)
    return RedirectResponse("/settings?connected=google")


# ── Recall.ai webhook ──────────────────────────────────────────────────────────

def _verify_recall_signature(request_body: bytes, headers: Dict[str, str], secret: str) -> bool:
    msg_id = headers.get("webhook-id") or headers.get("svix-id")
    msg_timestamp = headers.get("webhook-timestamp") or headers.get("svix-timestamp")
    msg_signature = headers.get("webhook-signature") or headers.get("svix-signature")
    if not secret.startswith("whsec_") or not msg_id or not msg_timestamp or not msg_signature:
        return False

    try:
        key = base64.b64decode(secret.removeprefix("whsec_"))
        payload = request_body.decode("utf-8")
    except Exception:
        return False

    signed_content = f"{msg_id}.{msg_timestamp}.{payload}".encode("utf-8")
    expected = base64.b64encode(
        hmac.new(key, signed_content, hashlib.sha256).digest()
    ).decode("utf-8")

    for versioned_signature in msg_signature.split(" "):
        try:
            version, signature = versioned_signature.split(",", 1)
        except ValueError:
            continue
        if version == "v1" and hmac.compare_digest(expected, signature):
            return True
    return False


@app.post("/webhook/recall/{meeting_id}")
async def recall_webhook(meeting_id: str, request: Request, payload: Dict[str, Any]):
    settings = get_settings()
    if settings.recall_webhook_secret:
        body = await request.body()
        headers = {key.lower(): value for key, value in request.headers.items()}
        if not _verify_recall_signature(body, headers, settings.recall_webhook_secret):
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
