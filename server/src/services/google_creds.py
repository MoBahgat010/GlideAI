import logging
from datetime import datetime, timezone
from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_TOKEN_URI,
    GMAIL_SCOPES,
)
from src.db.mongo import mongo

logger = logging.getLogger("server.services.google_creds")


def _get_token_doc(user_id: str) -> Optional[dict]:
    """Fetch the user's gmail_tokens document from MongoDB."""
    user = mongo.users.find_one({"_id": user_id})
    if not user:
        logger.warning("User not found for user_id=%s in google_creds", user_id)
        return None
    return user.get("gmail_tokens") or user.get("google_tokens")

async def get_user_google_credentials(user_id: str) -> Optional[Credentials]:
    """
    Retrieve Google OAuth credentials for a user, auto-refreshing if expired.
    Uses configured Google Client ID, Secret, Token URI, and Scopes.
    """
    token_doc = _get_token_doc(user_id)
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
        token_uri=GOOGLE_TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=scopes,
        expiry=expiry,
    )

    if creds.expired and creds.refresh_token:
        try:
            logger.info("Refreshing Google token for user_id=%s", user_id)
            creds.refresh(Request())
            mongo.users.update_one(
                {"_id": user_id},
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
