from datetime import datetime, timezone
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from src.auth.dependencies import get_current_user
from src.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_and_rotate_refresh_token,
    revoke_refresh_token,
    blacklist_access_token,
    hash_password,
    verify_password,
)
from src.db.mongo import mongo
from src.models.schemas import (
    Token,
    UserRegister,
    UserResponse,
    RefreshTokenRequest,
    LogoutRequest,
)

logger = logging.getLogger("server.routers.auth")
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister):
    existing_user = mongo.users.find_one({
        "$or": [{"username": user_in.username.strip()}, {"email": user_in.email.strip().lower()}]
    })
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered.",
        )

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    user_doc = {
        "_id": user_id,
        "username": user_in.username.strip(),
        "email": user_in.email.strip().lower(),
        "password_hash": hash_password(user_in.password),
        "created_at": now,
    }
    mongo.users.insert_one(user_doc)
    logger.info("Registered new user: %s (id=%s)", user_doc["username"], user_id)

    access_token = create_access_token(data={"sub": user_doc["username"], "user_id": user_id})
    refresh_token, ttl_seconds = await create_refresh_token(user_id=user_id, username=user_doc["username"])

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
        username=user_doc["username"],
        user_id=user_id,
    )


@router.post("/login", response_model=Token)
async def login(request: Request):
    identifier = ""
    password = ""

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            identifier = (body.get("username") or body.get("email") or "").strip()
            password = body.get("password") or ""
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            identifier = (form.get("username") or form.get("email") or "").strip()
            password = form.get("password") or ""
        except Exception:
            pass

    if not identifier or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username/email and password are required.",
        )

    user = mongo.users.find_one({
        "$or": [
            {"username": identifier},
            {"email": identifier.lower()},
        ]
    })
    if not user or not user.get("password_hash") or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user["_id"])
    username = str(user["username"])
    access_token = create_access_token(data={"sub": username, "user_id": user_id})
    refresh_token, ttl_seconds = await create_refresh_token(user_id=user_id, username=username)

    logger.info("User logged in: %s", username)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
        username=username,
        user_id=user_id,
    )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(payload: RefreshTokenRequest):
    result = await verify_and_rotate_refresh_token(payload.refresh_token)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id, username, new_refresh_token = result
    new_access_token = create_access_token(data={"sub": username, "user_id": user_id})

    logger.info("Refreshed access token for user: %s", username)
    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=900,
        username=username,
        user_id=user_id,
    )


@router.post("/logout")
async def logout(
    payload: Optional[LogoutRequest] = None,
    authorization: Optional[str] = Header(None),
    current_user: dict = Depends(get_current_user),
):
    if payload and payload.refresh_token:
        await revoke_refresh_token(payload.refresh_token)

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        await blacklist_access_token(token)

    logger.info("User logged out: %s", current_user.get("username"))
    return {"message": "Successfully logged out. Refresh token revoked."}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user["_id"]),
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user.get("created_at"),
    )
