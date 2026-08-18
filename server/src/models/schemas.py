from datetime import datetime
from typing import List, Optional, Any
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
    expires_in: int = 900
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


class FileMetadata(BaseModel):
    filename: str
    size: int = 0
    file_type: str = "document"
    file_url: Optional[str] = None
    url: Optional[str] = None
    uploaded_at: Optional[datetime] = None


class SessionCreate(BaseModel):
    title: str = Field(default="New RAG Session", description="Title of the conversation session")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    status: str
    created_at: datetime
    memory_extracted: bool = False
    files: Optional[List[FileMetadata]] = []


class SessionEndResponse(BaseModel):
    session_id: str
    status: str
    task_id: str
    message: str


class ChunkMetadata(BaseModel):
    index: Optional[int] = None
    custom_id: Optional[str] = None
    file_name: Optional[str] = None
    page: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    bbox: Optional[List[float]] = None
    type: Optional[str] = "text"
    score: Optional[float] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    file_url: Optional[str] = None
    url: Optional[str] = None


class Citation(ChunkMetadata):
    pass


class AnswerWithCitations(BaseModel):
    answer: str
    citations: List[ChunkMetadata] = []


class AgentAnswersResponse(BaseModel):
    session_id: str
    query: str
    answers: List[AnswerWithCitations] = []


class GenerateTitleRequest(BaseModel):
    prompt: str


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


class GoogleServiceStatus(BaseModel):
    connected: bool
    email: Optional[str] = None


class GmailMessageSummary(BaseModel):
    id: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    date: Optional[str] = None
    snippet: Optional[str] = None
    is_unread: bool = False


class GmailMessageDetail(GmailMessageSummary):
    body_text: Optional[str] = None
    to: Optional[str] = None


class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    reply_to_message_id: Optional[str] = None


class CreateDraftRequest(BaseModel):
    to: str
    subject: str
    body: str


class HiTLDecision(BaseModel):
    type: str                              # approve | edit | reject | respond
    message: Optional[str] = None         # for reject / respond
    edited_action: Optional[dict] = None  # for edit: {name, args}


class HiTLResumeRequest(BaseModel):
    decisions: List[HiTLDecision]

