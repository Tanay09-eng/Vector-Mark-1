"""
Vector AI — Pydantic Schemas
Request/Response models for all FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    use_voice: bool = False

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str = ""          # what the orchestrator detected
    tool_used: str = ""       # which tool was called, if any
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Tasks ─────────────────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    title: str
    description: str = ""
    project: str = "General"
    priority: str = "normal"
    due_date: Optional[datetime] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    project: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[datetime] = None

class TaskOut(BaseModel):
    id: int
    title: str
    description: str
    project: str
    priority: str
    status: str
    due_date: Optional[datetime]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ── Reminders ─────────────────────────────────────────────────────────────────

class ReminderCreate(BaseModel):
    title: str
    description: str = ""
    remind_at: datetime
    repeat: str = "none"

class ReminderOut(BaseModel):
    id: int
    title: str
    description: str
    remind_at: datetime
    repeat: str
    fired: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Progress ──────────────────────────────────────────────────────────────────

class ProgressCreate(BaseModel):
    project: str
    entry: str
    notes: str = ""
    mood: str = ""

class ProgressOut(BaseModel):
    id: int
    project: str
    entry: str
    notes: str
    mood: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Memory ────────────────────────────────────────────────────────────────────

class MemoryOut(BaseModel):
    id: int
    content: str
    source: str
    importance: float
    created_at: datetime

    class Config:
        from_attributes = True


# ── Calendar ──────────────────────────────────────────────────────────────────

class CalendarEventOut(BaseModel):
    id: int
    google_id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str

    class Config:
        from_attributes = True


# ── System ────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    ollama_online: bool
    db_online: bool
    tools: dict
    version: str = "2.0.0"
