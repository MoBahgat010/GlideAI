import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Any
from bson import ObjectId

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_TOKEN_URI,
    GMAIL_SCOPES,
)
from src.db.mongo import MongoManager, mongo_db

logger = logging.getLogger("server.services.google_creds")


async def _get_user_doc(user_id: str) -> Optional[dict]:
    """Find user document by ObjectId, string _id, username, or email."""
    if not user_id:
        return None

    db = MongoManager.get_db()
    
    # 1. Try ObjectId match
    if ObjectId.is_valid(user_id):
        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if user:
                return user
        except Exception:
            pass

    # 2. Try string _id
    user = await db.users.find_one({"_id": user_id})
    if user:
        return user

    # 3. Try username or email
    user = await db.users.find_one({"username": user_id})
    if user:
        return user

    user = await db.users.find_one({"email": user_id})
    return user


async def _get_token_doc(user_id: str) -> Tuple[Optional[dict], Optional[Any]]:
    """Fetch the user's gmail_tokens document and user _id from MongoDB."""
    user = await _get_user_doc(user_id)
    if not user:
        logger.warning("User not found for user_id=%s in google_creds", user_id)
        return None, None

    token_doc = user.get("gmail_tokens") or user.get("google_tokens")
    if not token_doc:
        logger.info("User %s has no Google tokens stored.", user.get("username", user_id))
        return None, user.get("_id")

    return token_doc, user.get("_id")


async def get_user_refresh_token(user_id: str) -> Optional[str]:
    """Return the stored Gmail refresh_token for a user."""
    token_doc, _ = await _get_token_doc(user_id)
    if not token_doc:
        return None
    return token_doc.get("refresh_token")


async def get_user_google_credentials(user_id: str) -> Optional[Credentials]:
    """
    Retrieve Google OAuth credentials for a user, auto-refreshing if expired.
    Uses configured Google Client ID, Secret, Token URI, and Scopes.
    """
    token_doc, real_user_id = await _get_token_doc(user_id)
    if not token_doc:
        return None

    access_token = token_doc.get("access_token")
    refresh_token = token_doc.get("refresh_token")
    if not access_token and not refresh_token:
        logger.warning("No access_token or refresh_token found in token doc for user_id=%s", user_id)
        return None

    expiry = token_doc.get("expiry")
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry)
        except Exception:
            expiry = None

    # Normalize to offset-naive UTC so google-auth internal comparison works
    if expiry and expiry.tzinfo is not None:
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)

    scopes = token_doc.get("scope")
    if isinstance(scopes, str):
        scopes = scopes.split()
    elif not scopes:
        scopes = GMAIL_SCOPES

    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri=GOOGLE_TOKEN_URI or "https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=scopes,
        expiry=expiry,
    )

    if creds.expired and creds.refresh_token:
        try:
            logger.info("Refreshing Google token for user_id=%s (real_id=%s)", user_id, real_user_id)
            creds.refresh(GoogleRequest())
            db = MongoManager.get_db()
            update_filter = {"_id": real_user_id} if real_user_id is not None else {"_id": user_id}
            await db.users.update_one(
                update_filter,
                {"$set": {
                    "gmail_tokens.access_token": creds.token,
                    "gmail_tokens.expiry": creds.expiry.isoformat() if creds.expiry else None,
                    **({"gmail_tokens.refresh_token": creds.refresh_token} if creds.refresh_token else {}),
                }},
            )
            logger.info("Token refreshed and saved successfully for user_id=%s", user_id)
        except Exception as exc:
            logger.error("Failed to refresh Google credentials for user_id=%s: %s", user_id, exc)
            return None

    return creds
