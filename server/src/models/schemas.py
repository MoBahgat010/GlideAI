from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds
    username: str
    user_id: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: Optional[datetime] = None


class SessionCreate(BaseModel):
    title: str = Field(default="New RAG Session", description="Title of the conversation session")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    status: str
    created_at: datetime
    memory_extracted: bool = False


class SessionEndResponse(BaseModel):
    session_id: str
    status: str
    task_id: str
    message: str


class Citation(BaseModel):
    custom_id: str
    file_name: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    type: Optional[str] = "text"
    score: Optional[float] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    cloudinary_url: Optional[str] = None


class AskRequest(BaseModel):
    query: str


class AskSessionRequest(BaseModel):
    query: str


class EpisodicMemoryResponse(BaseModel):
    session_id: str
    summary: str
    key_events: List[str]
    created_at: Optional[datetime] = None


class SemanticMemoryResponse(BaseModel):
    session_id: str
    facts: List[str]
    preferences: List[str]
    created_at: Optional[datetime] = None


class MemoryStatusResponse(BaseModel):
    session_id: str
    enable_semantic: bool
    enable_episodic: bool
    enable_working: bool
    episodic_memory: Optional[dict] = None
    semantic_memory: Optional[dict] = None
