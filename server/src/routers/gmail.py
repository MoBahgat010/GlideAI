import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from config import (
    GOOGLE_AUTH_URI,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_TOKEN_URI,
    GMAIL_REDIRECT_URI,
    FRONTEND_URL,
    GMAIL_SCOPES,
)
from src.auth.dependencies import get_current_user, get_optional_user
from src.db.mongo import MongoManager
from src.db.redis import RedisManager
from src.models.schemas import GoogleServiceStatus
from src.services.google_creds import get_user_google_credentials

logger = logging.getLogger("server.routers.gmail")
router = APIRouter(prefix="/api/gmail", tags=["Gmail"])

STATE_TTL = 300


@router.get("/connect")
async def gmail_connect(
    token: Optional[str] = Query(None, description="JWT token for browser redirect"),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Start Google Gmail OAuth flow."""
    user = current_user
    if not user and token:
        user = await get_current_user(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    user_id = str(user["_id"])
    state = str(uuid.uuid4())
    redis = RedisManager.get_client()
    await redis.setex(f"gmail_oauth_state:{state}", STATE_TTL, user_id)

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GMAIL_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GMAIL_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{GOOGLE_AUTH_URI or 'https://accounts.google.com/o/oauth2/auth'}?{urlencode(params)}"
    return RedirectResponse(auth_url)


@router.get("/callback")
async def gmail_callback(code: str, state: str):
    """Handle Gmail OAuth callback and persist tokens to MongoDB."""
    redis = RedisManager.get_client()
    state_key = f"gmail_oauth_state:{state}"
    user_id = await redis.get(state_key)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OAuth state.")
    await redis.delete(state_key)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URI or "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GMAIL_REDIRECT_URI,
            },
        )
        if resp.status_code != 200:
            logger.error("Gmail token exchange failed: %s %s", resp.status_code, resp.text)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google token exchange failed.")
        tokens = resp.json()

    now = datetime.now(timezone.utc)
    expiry = datetime.fromtimestamp(now.timestamp() + tokens.get("expires_in", 3600), timezone.utc).isoformat()

    token_doc = {
        "access_token": tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "expiry": expiry,
        "scope": tokens.get("scope", " ".join(GMAIL_SCOPES)),
    }

    # Preserve existing refresh_token if Google didn't return a new one
    if not token_doc["refresh_token"]:
        db = MongoManager.get_db()
        existing = await db.users.find_one({"_id": user_id})
        if existing and existing.get("gmail_tokens"):
            token_doc["refresh_token"] = existing["gmail_tokens"].get("refresh_token")

    db = MongoManager.get_db()
    await db.users.update_one({"_id": user_id}, {"$set": {"gmail_tokens": token_doc}})
    logger.info("Gmail tokens stored for user_id=%s", user_id)
    return RedirectResponse(f"{FRONTEND_URL}?gmail_connected=true")


@router.get("/status", response_model=GoogleServiceStatus)
async def gmail_status(current_user: dict = Depends(get_current_user)):
    """Return whether the current user has Gmail connected."""
    token_doc = current_user.get("gmail_tokens")
    if not token_doc or not token_doc.get("refresh_token"):
        return GoogleServiceStatus(connected=False)

    # Optionally resolve the connected email via credentials
    try:
        from googleapiclient.discovery import build
        creds = await get_user_google_credentials(str(current_user["_id"]))
        if creds:
            service = build("gmail", "v1", credentials=creds)
            profile = service.users().getProfile(userId="me").execute()
            return GoogleServiceStatus(connected=True, email=profile.get("emailAddress"))
    except Exception:
        pass

    return GoogleServiceStatus(connected=True)
