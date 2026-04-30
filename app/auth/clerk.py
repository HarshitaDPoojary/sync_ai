import logging
import time
import uuid

import httpx
from fastapi import HTTPException, Request
from jose import JWTError, jwt

from app.core.config import get_settings

logger = logging.getLogger("sync_ai.auth")

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}


async def _get_jwks() -> dict:
    settings = get_settings()
    now = time.time()
    if _jwks_cache["keys"] and now - _jwks_cache["fetched_at"] < 3600:
        return _jwks_cache["keys"]
    url = f"https://{settings.clerk_frontend_api}/.well-known/jwks.json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json()
        _jwks_cache["fetched_at"] = now
    return _jwks_cache["keys"]


async def get_current_user(request: Request):
    from app.repositories.user_repo import UserRepository

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth[7:]

    try:
        jwks = await _get_jwks()
        claims = jwt.decode(token, jwks, algorithms=["RS256"])
    except JWTError as exc:
        logger.warning("clerk_jwt_invalid error=%s", exc)
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as exc:
        logger.error("clerk_jwks_fetch_failed error=%s", exc)
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    clerk_user_id: str = claims["sub"]
    repo = UserRepository()
    user = repo.get_by_clerk_id(clerk_user_id)
    if not user:
        user = repo.create(
            clerk_user_id=clerk_user_id,
            email=claims.get("email", ""),
            name=claims.get("name", ""),
        )
        logger.info("user_provisioned clerk_user_id=%s email=%s", clerk_user_id, user.email)
    return user
