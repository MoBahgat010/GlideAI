from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from server.src.auth.jwt import decode_access_token
from server.src.db.mongo import MongoManager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    """Dependency verifying JWT token and returning authenticated user document."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload["sub"]
    db = MongoManager.get_db()
    user = await db.users.find_one({"username": user_id})
    if not user:
        # Try finding by string ID
        user = await db.users.find_one({"_id": user_id})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def get_optional_user(token: str | None = Depends(oauth2_scheme)) -> dict | None:
    """Optional user dependency — returns user document if token valid, else None."""
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
