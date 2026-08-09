from datetime import datetime
from pydantic import BaseModel, Field


# ── Auth Schemas ───────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime | None = None


# ── Session Schemas ────────────────────────────────────────────────────────────

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


# ── Chat / RAG Schemas ─────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str


class AskSessionRequest(BaseModel):
    query: str


# ── Memory Schemas ─────────────────────────────────────────────────────────────

class EpisodicMemoryResponse(BaseModel):
    session_id: str
    summary: str
    key_events: list[str]
    created_at: datetime | None = None


class SemanticMemoryResponse(BaseModel):
    session_id: str
    facts: list[str]
    preferences: list[str]
    created_at: datetime | None = None
