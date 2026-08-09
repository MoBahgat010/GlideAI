from datetime import datetime, timedelta
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase

from server.src.auth.dependencies import get_current_user
from server.src.auth.jwt import create_access_token, hash_password, verify_password
from config import JWT_ACCESS_TOKEN_EXPIRE_MINUTES

from server.src.db.mongo import get_database
from server.src.models.schemas import Token, UserLogin, UserRegister, UserResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister, db: AsyncIOMotorDatabase = Depends(get_database)):
    """Register a new user in MongoDB."""
    existing_user = await db.users.find_one({
        "$or": [{"username": user_in.username}, {"email": user_in.email}]
    })
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered.",
        )

    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "username": user_in.username,
        "email": user_in.email,
        "password_hash": hash_password(user_in.password),
        "created_at": datetime.utcnow(),
    }
    await db.users.insert_one(user_doc)

    return UserResponse(
        id=user_id,
        username=user_in.username,
        email=user_in.email,
        created_at=user_doc["created_at"],
    )


@router.post("/login", response_model=Token)
async def login(
    credentials: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Authenticate user with username and password, returning JWT token."""
    user = await db.users.find_one({"username": credentials.username})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user["username"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires,
    )
    return Token(access_token=token, username=user["username"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=str(current_user["_id"]),
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user.get("created_at"),
    )
