"""
Vector AI — Database Layer
SQLite via SQLAlchemy ORM
Tables: users, messages, tasks, reminders, progress, memory_chunks
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Boolean, DateTime, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

BASE_DIR = os.path.join(os.path.expanduser("~"), ".vector_ai")
os.makedirs(BASE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{BASE_DIR}/vector.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Message(Base):
    """Conversation history — every user/assistant turn."""
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, index=True)
    role       = Column(String(16))          # "user" | "assistant" | "system"
    content    = Column(Text)
    session_id = Column(String(64))          # groups a conversation session
    timestamp  = Column(DateTime, default=datetime.utcnow)
    tokens     = Column(Integer, default=0)  # rough token count


class Task(Base):
    """To-do items."""
    __tablename__ = "tasks"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(256))
    description = Column(Text, default="")
    project     = Column(String(128), default="General")
    priority    = Column(String(16), default="normal")   # low|normal|high|urgent
    status      = Column(String(16), default="pending")  # pending|done|cancelled
    due_date    = Column(DateTime, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    completed_at= Column(DateTime, nullable=True)


class Reminder(Base):
    """Time-based reminders."""
    __tablename__ = "reminders"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(256))
    description = Column(Text, default="")
    remind_at   = Column(DateTime)
    repeat      = Column(String(32), default="none")  # none|daily|weekly
    fired       = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class ProgressEntry(Base):
    """Work progress log (McKinsey Forward, college, etc.)."""
    __tablename__ = "progress"

    id         = Column(Integer, primary_key=True, index=True)
    project    = Column(String(128))
    entry      = Column(Text)
    notes      = Column(Text, default="")
    mood       = Column(String(32), default="")   # optional mood tag
    created_at = Column(DateTime, default=datetime.utcnow)


class MemoryChunk(Base):
    """
    Semantic memory — key facts Vector remembers about the user.
    Each chunk has an embedding vector stored as a JSON string.
    """
    __tablename__ = "memory_chunks"

    id         = Column(Integer, primary_key=True, index=True)
    content    = Column(Text)                  # the fact/summary
    source     = Column(String(32))            # "conversation"|"user"|"progress"
    embedding  = Column(Text, default="")      # JSON list of floats
    importance = Column(Float, default=1.0)    # higher = retrieved more often
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class CalendarEvent(Base):
    """Cached Google Calendar events."""
    __tablename__ = "calendar_events"

    id           = Column(Integer, primary_key=True, index=True)
    google_id    = Column(String(256), unique=True)
    title        = Column(String(256))
    description  = Column(Text, default="")
    start_time   = Column(DateTime)
    end_time     = Column(DateTime)
    location     = Column(String(256), default="")
    synced_at    = Column(DateTime, default=datetime.utcnow)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Database ready at {BASE_DIR}/vector.db")


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
