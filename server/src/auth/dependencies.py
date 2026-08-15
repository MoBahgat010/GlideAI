import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from .jwt import decode_access_token
from ..db.mongo import MongoManager

logger = logging.getLogger("server.auth.dependencies")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
        )
    payload = await decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )

    username = payload["sub"]
    db = MongoManager.get_db()
    user = await db.users.find_one({"username": username})
    if not user:
        user_id = payload.get("user_id")
        if user_id:
            user = await db.users.find_one({"_id": user_id})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """Optional user dependency — returns user document if valid token present, else None."""
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
