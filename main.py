"""
Vector AI — FastAPI Backend Server
The central hub. Connects all layers:
  Database → Memory → LLM → Orchestrator → Tools → Voice

Run: uvicorn main:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "voice"))

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional
import asyncio
import tempfile
import json

from database import init_db, get_db, Task, Reminder, ProgressEntry, MemoryChunk
from schemas import (
    ChatRequest, ChatResponse, TaskCreate, TaskUpdate, TaskOut,
    ReminderCreate, ReminderOut, ProgressCreate, ProgressOut,
    MemoryOut, StatusResponse
)
from orchestrator import Orchestrator
from llm import llm
from tools import NewsTool, CalendarTool

# ── Load config ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.expanduser("~"), ".vector_ai")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            return json.loads(open(CONFIG_FILE).read())
        except Exception:
            return {}
    return {}

config = load_config()
NEWS_API_KEY = "5e78eef3c5b74ce5bd8f9af6aeab91af"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vector AI",
    description="Personal AI Assistant — J.A.R.V.I.S style",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Frontend (Electron/HTML) on any port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    print("\n" + "═"*50)
    print("  VECTOR AI — Backend Online")
    print(f"  LLM     : {'Ollama ✓' if llm.is_online() else 'Ollama Offline — run: ollama serve'}")
    print(f"  News    : {'Configured ✓' if NEWS_API_KEY else 'Not configured'}")
    print("  Docs    : http://localhost:8000/docs")
    print("═"*50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CHAT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.
    The Orchestrator decides everything from here.
    """
    orchestrator = Orchestrator(db, news_api_key=NEWS_API_KEY)
    result = orchestrator.process(
        user_input=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(
        reply=result["reply"],
        session_id=request.session_id,
        intent=result["intent"],
        tool_used=result["tool_used"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/voice/speak")
async def speak(text: str, voice: str = "en-GB-RyanNeural"):
    """
    Convert text to speech using Edge TTS.
    Returns an MP3 audio stream.
    """
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        await communicate.save(temp_path)
        return FileResponse(
            temp_path,
            media_type="audio/mpeg",
            filename="vector_response.mp3",
        )
    except ImportError:
        raise HTTPException(503, "edge-tts not installed. Run: pip install edge-tts")
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}")


@app.get("/voice/voices")
async def list_voices():
    """List all available Edge TTS voices."""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        english = [v for v in voices if v["Locale"].startswith("en")]
        return {"voices": english}
    except ImportError:
        raise HTTPException(503, "edge-tts not installed.")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/tasks", response_model=List[TaskOut])
def get_tasks(status: str = "pending", db: Session = Depends(get_db)):
    q = db.query(Task)
    if status != "all":
        q = q.filter(Task.status == status)
    return q.order_by(Task.created_at.desc()).all()

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    db_task = Task(**task.model_dump(), status="pending", created_at=datetime.utcnow())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.patch("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, update: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")
    for field, value in update.model_dump(exclude_none=True).items():
        setattr(task, field, value)
    if update.status == "done":
        task.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return task

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")
    db.delete(task)
    db.commit()
    return {"message": f"Task {task_id} deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# REMINDER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/reminders", response_model=List[ReminderOut])
def get_reminders(db: Session = Depends(get_db)):
    return (db.query(Reminder)
            .filter(Reminder.fired == False)
            .order_by(Reminder.remind_at)
            .all())

@app.post("/reminders", response_model=ReminderOut)
def create_reminder(reminder: ReminderCreate, db: Session = Depends(get_db)):
    db_r = Reminder(**reminder.model_dump(), fired=False, created_at=datetime.utcnow())
    db.add(db_r)
    db.commit()
    db.refresh(db_r)
    return db_r

@app.delete("/reminders/{reminder_id}")
def delete_reminder(reminder_id: int, db: Session = Depends(get_db)):
    r = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not r:
        raise HTTPException(404, "Reminder not found")
    db.delete(r)
    db.commit()
    return {"message": "Reminder deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/progress", response_model=List[ProgressOut])
def get_progress(project: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(ProgressEntry)
    if project:
        q = q.filter(ProgressEntry.project.ilike(f"%{project}%"))
    return q.order_by(ProgressEntry.created_at.desc()).limit(50).all()

@app.post("/progress", response_model=ProgressOut)
def log_progress(entry: ProgressCreate, db: Session = Depends(get_db)):
    db_p = ProgressEntry(**entry.model_dump(), created_at=datetime.utcnow())
    db.add(db_p)
    db.commit()
    db.refresh(db_p)
    return db_p


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/memory", response_model=List[MemoryOut])
def get_memories(db: Session = Depends(get_db)):
    return db.query(MemoryChunk).order_by(MemoryChunk.importance.desc()).limit(20).all()

@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: int, db: Session = Depends(get_db)):
    m = db.query(MemoryChunk).filter(MemoryChunk.id == memory_id).first()
    if not m:
        raise HTTPException(404, "Memory not found")
    db.delete(m)
    db.commit()
    return {"message": "Memory deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/calendar/today")
def calendar_today(db: Session = Depends(get_db)):
    cal = CalendarTool(db)
    return {"result": cal.today()}

@app.get("/calendar/upcoming")
def calendar_upcoming(days: int = 7, db: Session = Depends(get_db)):
    cal = CalendarTool(db)
    return {"result": cal.upcoming(days=days)}


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/news")
def get_news(query: Optional[str] = None, country: str = "in"):
    news = NewsTool(api_key=NEWS_API_KEY)
    if query:
        return {"result": news.search(query)}
    return {"result": news.top_headlines(country=country)}


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/status", response_model=StatusResponse)
def system_status(db: Session = Depends(get_db)):
    cal = CalendarTool(db)
    return StatusResponse(
        status="online",
        ollama_online=llm.is_online(),
        db_online=True,
        tools={
            "news": bool(NEWS_API_KEY),
            "calendar": cal.is_available(),
            "search": True,
            "tasks": True,
            "reminders": True,
            "progress": True,
        }
    )

@app.get("/config")
def get_config():
    """Return non-sensitive config info."""
    return {
        "ollama_online": llm.is_online(),
        "ollama_models": llm.available_models(),
        "version": "2.0.0",
    }

@app.post("/config/save")
def save_config_endpoint(news_key: str = "", gemini_key: str = ""):
    """Save API keys to local config file."""
    cfg = load_config()
    if news_key:
        cfg["news_api_key"] = news_key
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return {"message": "Config saved."}
