import uuid
import logging
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_AUTH_URI,
    GOOGLE_TOKEN_URI,
    GOOGLE_REDIRECT_URI,
    FRONTEND_URL,
    GMAIL_SCOPES,
)
from src.auth.jwt import create_access_token, create_refresh_token
from src.db.mongo import MongoManager
from src.db.redis import RedisManager

logger = logging.getLogger("server.routers.google_auth")
router = APIRouter(prefix="/api/auth", tags=["Google Auth"])

STATE_TTL = 300  # seconds

AUTH_SCOPES = [
    "openid",
    "email",
    "profile",
] + GMAIL_SCOPES


@router.get("/google/login")
async def google_login():
    """Redirect the browser to Google's OAuth2 authorization page."""
    state = str(uuid.uuid4())
    redis = RedisManager.get_client()
    await redis.setex(f"google_oauth_state:{state}", STATE_TTL, "1")

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(AUTH_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_base = GOOGLE_AUTH_URI or "https://accounts.google.com/o/oauth2/auth"
    auth_url = f"{auth_base}?{urlencode(params)}"
    return RedirectResponse(auth_url)


@router.get("/google/callback")
async def google_callback(code: str, state: str):
    """Exchange the authorization code for tokens, upsert user in MongoDB with tokens, and redirect to frontend."""
    redis = RedisManager.get_client()
    state_key = f"google_oauth_state:{state}"
    stored = await redis.get(state_key)
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state.",
        )
    await redis.delete(state_key)

    # Exchange authorization code for tokens directly with Google token endpoint
    token_url = GOOGLE_TOKEN_URI or "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(token_url, data=token_data)
        if token_resp.status_code != 200:
            logger.error("Google token exchange failed: status=%s body=%s", token_resp.status_code, token_resp.text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Google token exchange failed: {token_resp.text}",
            )
        tokens = token_resp.json()

    # Retrieve user profile from ID token or userinfo endpoint
    raw_id_token = tokens.get("id_token")
    id_info = None

    if raw_id_token:
        try:
            id_info = id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                GOOGLE_CLIENT_ID,
            )
        except Exception as exc:
            logger.warning("ID token verification failed (%s), falling back to userinfo endpoint...", exc)

    if not id_info:
        # Fallback to userinfo endpoint using the access token
        access_token_val = tokens.get("access_token")
        async with httpx.AsyncClient() as client:
            u_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token_val}"},
            )
            if u_resp.status_code == 200:
                id_info = u_resp.json()
            else:
                logger.error("Failed to fetch userinfo: status=%s body=%s", u_resp.status_code, u_resp.text)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to retrieve Google user profile.",
                )

    google_id = id_info.get("sub")
    email = id_info.get("email", "")
    name = id_info.get("name", email.split("@")[0] if email else "GoogleUser")
    picture = id_info.get("picture")

    if not google_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google user ID.",
        )

    # Prepare OAuth token document for Workspace & Gmail operations
    token_doc = None
    if tokens.get("access_token"):
        now = datetime.now(timezone.utc)
        expires_in = tokens.get("expires_in", 3600)
        expiry = now.timestamp() + expires_in
        token_doc = {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expiry": datetime.fromtimestamp(expiry, timezone.utc).isoformat(),
            "scope": tokens.get("scope", " ".join(AUTH_SCOPES)),
        }

    # Upsert user in MongoDB
    db = MongoManager.get_db()
    user = await db.users.find_one({"google_id": google_id})

    if user:
        user_id = str(user["_id"])
        username = str(user["username"])
        update_fields = {"google_avatar": picture}
        if token_doc:
            if not token_doc.get("refresh_token") and user.get("gmail_tokens"):
                token_doc["refresh_token"] = user["gmail_tokens"].get("refresh_token")
            update_fields["gmail_tokens"] = token_doc
        await db.users.update_one({"_id": user["_id"]}, {"$set": update_fields})
    else:
        # Link by matching email if account already exists
        if email:
            user = await db.users.find_one({"email": email.lower()})
        if user:
            user_id = str(user["_id"])
            username = str(user["username"])
            update_fields = {"google_id": google_id, "google_avatar": picture}
            if token_doc:
                if not token_doc.get("refresh_token") and user.get("gmail_tokens"):
                    token_doc["refresh_token"] = user["gmail_tokens"].get("refresh_token")
                update_fields["gmail_tokens"] = token_doc
            await db.users.update_one({"_id": user["_id"]}, {"$set": update_fields})
        else:
            # Create a brand new user
            user_id = str(uuid.uuid4())
            base_uname = name.replace(" ", "_").lower()[:30]
            username = base_uname
            existing = await db.users.find_one({"username": username})
            if existing:
                username = f"{base_uname}_{uuid.uuid4().hex[:6]}"

            now = datetime.now(timezone.utc)
            await db.users.insert_one({
                "_id": user_id,
                "username": username,
                "email": email.lower() if email else "",
                "password_hash": None,
                "google_id": google_id,
                "google_avatar": picture,
                "created_at": now,
                "gmail_tokens": token_doc,
            })
            logger.info("Created new Google user: %s (id=%s) with stored tokens", username, user_id)

    # Issue JWT tokens
    access_token = create_access_token(data={"sub": username, "user_id": user_id})
    refresh_token, _ = await create_refresh_token(user_id=user_id, username=username)

    logger.info("Google sign-in successful: %s (tokens stored: %s)", username, bool(token_doc))
    redirect_url = (
        f"{FRONTEND_URL}/auth/callback"
        f"#access_token={access_token}"
        f"&refresh_token={refresh_token}"
        f"&username={username}"
    )
    return RedirectResponse(redirect_url)
