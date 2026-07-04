"""
Vector AI — Tools Layer
Each tool is a self-contained class.
The Orchestrator calls these based on detected intent.

Tools:
  - TaskTool       → CRUD on tasks in SQLite
  - ReminderTool   → Create/list/fire reminders
  - NewsTool       → NewsAPI headlines & search
  - SearchTool     → DuckDuckGo instant answers
  - CalendarTool   → Google Calendar via OAuth2
  - ProgressTool   → Work progress logging
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from database import Task, Reminder, ProgressEntry, CalendarEvent


# ═══════════════════════════════════════════════════════════════════════════════
# TASK TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class TaskTool:
    def __init__(self, db: Session):
        self.db = db

    def add(self, title: str, project: str = "General",
            priority: str = "normal", description: str = "",
            due_date: datetime = None) -> str:
        task = Task(
            title=title, description=description,
            project=project, priority=priority,
            status="pending", due_date=due_date,
            created_at=datetime.utcnow(),
        )
        self.db.add(task)
        self.db.commit()
        return f"Task added: '{title}' [{priority.upper()}] under {project}."

    def list_pending(self) -> List[Dict]:
        tasks = self.db.query(Task).filter(Task.status == "pending").all()
        return [
            {"id": t.id, "title": t.title, "project": t.project,
             "priority": t.priority, "due_date": str(t.due_date) if t.due_date else None}
            for t in tasks
        ]

    def complete(self, task_id: int) -> str:
        task = self.db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return f"No task found with ID {task_id}."
        task.status = "done"
        task.completed_at = datetime.utcnow()
        self.db.commit()
        return f"Task #{task_id} '{task.title}' marked complete. Well done, sir."

    def format_list(self) -> str:
        tasks = self.list_pending()
        if not tasks:
            return "Your task slate is clear, sir. No pending items."
        lines = ["Pending Tasks:"]
        for t in tasks:
            due = f" (due {t['due_date'][:10]})" if t["due_date"] else ""
            lines.append(f"  [{t['id']}] {t['title']} — {t['project']} [{t['priority'].upper()}]{due}")
        return "\n".join(lines)

    def parse_and_add(self, user_input: str) -> str:
        """
        Parse natural language task input.
        E.g. "Add task: review McKinsey module 3 for tomorrow high priority"
        """
        text = user_input.lower()
        priority = "normal"
        if "urgent" in text or "asap" in text:
            priority = "urgent"
        elif "high" in text:
            priority = "high"
        elif "low" in text:
            priority = "low"

        project = "General"
        for keyword in ["mckinsey", "college", "work", "personal", "forward"]:
            if keyword in text:
                project = keyword.title()
                break

        # Strip command words to get the actual task title
        title = user_input
        for strip in ["add task:", "add task", "add a task", "create task",
                      "new task", "task:"]:
            title = title.replace(strip, "").strip()

        return self.add(title=title or user_input, project=project, priority=priority)


# ═══════════════════════════════════════════════════════════════════════════════
# REMINDER TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class ReminderTool:
    def __init__(self, db: Session):
        self.db = db

    def add(self, title: str, remind_at: datetime,
            description: str = "", repeat: str = "none") -> str:
        reminder = Reminder(
            title=title, description=description,
            remind_at=remind_at, repeat=repeat,
            fired=False, created_at=datetime.utcnow(),
        )
        self.db.add(reminder)
        self.db.commit()
        return f"Reminder set: '{title}' at {remind_at.strftime('%b %d, %Y %H:%M')}."

    def list_upcoming(self) -> str:
        now = datetime.utcnow()
        reminders = (
            self.db.query(Reminder)
            .filter(Reminder.remind_at >= now, Reminder.fired == False)
            .order_by(Reminder.remind_at)
            .limit(10)
            .all()
        )
        if not reminders:
            return "No upcoming reminders."
        lines = ["Upcoming Reminders:"]
        for r in reminders:
            lines.append(f"  [{r.id}] {r.title} — {r.remind_at.strftime('%b %d %H:%M')} ({r.repeat})")
        return "\n".join(lines)

    def check_due(self) -> List[Dict]:
        """Called by the background scheduler to find fired reminders."""
        now = datetime.utcnow()
        due = (
            self.db.query(Reminder)
            .filter(Reminder.remind_at <= now, Reminder.fired == False)
            .all()
        )
        fired = []
        for r in due:
            fired.append({"id": r.id, "title": r.title, "description": r.description})
            if r.repeat == "none":
                r.fired = True
            elif r.repeat == "daily":
                r.remind_at = r.remind_at + timedelta(days=1)
            elif r.repeat == "weekly":
                r.remind_at = r.remind_at + timedelta(weeks=1)
        self.db.commit()
        return fired

    def parse_time(self, text: str) -> Optional[datetime]:
        """Very simple natural language time parser."""
        now = datetime.now()
        text = text.lower()
        if "tomorrow" in text:
            return now.replace(hour=9, minute=0) + timedelta(days=1)
        if "tonight" in text:
            return now.replace(hour=20, minute=0)
        if "in 1 hour" in text or "in an hour" in text:
            return now + timedelta(hours=1)
        if "in 30 minutes" in text or "in half an hour" in text:
            return now + timedelta(minutes=30)
        # default to 1 hour from now
        return now + timedelta(hours=1)


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class NewsTool:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "")

    def top_headlines(self, country: str = "in", count: int = 5) -> str:
        if not self.api_key:
            return "NewsAPI key not configured. Add NEWS_API_KEY to your .env file."
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": country, "pageSize": count, "apiKey": self.api_key},
                timeout=8,
            )
            data = r.json()
            if data.get("status") != "ok":
                return f"NewsAPI error: {data.get('message')}"
            articles = data.get("articles", [])
            if not articles:
                return "No headlines found."
            lines = [f"Top Headlines ({country.upper()}):"]
            for i, a in enumerate(articles, 1):
                lines.append(f"  {i}. {a['title']}")
                lines.append(f"     {a['source']['name']} · {a['publishedAt'][:10]}")
            return "\n".join(lines)
        except Exception as e:
            return f"News fetch failed: {e}"

    def search(self, query: str, count: int = 5) -> str:
        if not self.api_key:
            return "NewsAPI key not configured."
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query, "sortBy": "publishedAt",
                    "pageSize": count, "language": "en",
                    "apiKey": self.api_key,
                },
                timeout=8,
            )
            data = r.json()
            if data.get("status") != "ok":
                return f"NewsAPI error: {data.get('message')}"
            articles = data.get("articles", [])
            if not articles:
                return f"No articles found for '{query}'."
            lines = [f"News — {query.title()}:"]
            for i, a in enumerate(articles, 1):
                lines.append(f"  {i}. {a['title']}")
                lines.append(f"     {a['source']['name']} · {a['publishedAt'][:10]}")
                lines.append(f"     {a.get('url', '')}")
            return "\n".join(lines)
        except Exception as e:
            return f"News search failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class SearchTool:
    def search(self, query: str) -> str:
        try:
            r = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=8,
            )
            data = r.json()
            parts = []
            if data.get("Answer"):
                parts.append(f"Answer: {data['Answer']}")
            if data.get("AbstractText"):
                parts.append(f"Summary: {data['AbstractText']}")
            if data.get("Definition"):
                parts.append(f"Definition: {data['Definition']}")
            related = [t["Text"] for t in data.get("RelatedTopics", [])[:3] if "Text" in t]
            if related:
                parts.append("Related:\n" + "\n".join(f"  • {r}" for r in related))
            if not parts:
                return f"No instant answer found for '{query}'."
            return f"Search — {query}:\n" + "\n\n".join(parts)
        except Exception as e:
            return f"Search failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressTool:
    def __init__(self, db: Session):
        self.db = db

    def log(self, project: str, entry: str, notes: str = "", mood: str = "") -> str:
        p = ProgressEntry(
            project=project, entry=entry,
            notes=notes, mood=mood,
            created_at=datetime.utcnow(),
        )
        self.db.add(p)
        self.db.commit()
        return f"Progress logged under '{project}': {entry}"

    def summary(self, project: str = None, limit: int = 10) -> str:
        q = self.db.query(ProgressEntry)
        if project:
            q = q.filter(ProgressEntry.project.ilike(f"%{project}%"))
        entries = q.order_by(ProgressEntry.created_at.desc()).limit(limit).all()
        if not entries:
            return f"No progress entries found{' for ' + project if project else ''}."
        lines = [f"Progress Log{' — ' + project if project else ''}:"]
        for e in entries:
            lines.append(f"  [{e.created_at.strftime('%b %d')}] {e.project}: {e.entry}")
            if e.notes:
                lines.append(f"    Notes: {e.notes}")
        return "\n".join(lines)

    def active_projects(self) -> List[str]:
        rows = self.db.query(ProgressEntry.project).distinct().all()
        return [r[0] for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE CALENDAR TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class CalendarTool:
    """
    Google Calendar integration via OAuth2.

    First-time setup:
    1. Go to https://console.cloud.google.com/
    2. Create a project → Enable "Google Calendar API"
    3. Create OAuth2 credentials (Desktop App) → download credentials.json
    4. Place credentials.json in ~/.vector_ai/
    5. Run Vector — it will open a browser to authorize on first run.

    After authorization, token.json is saved and reused automatically.
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".vector_ai", "token.json")
    CREDS_PATH = os.path.join(os.path.expanduser("~"), ".vector_ai", "credentials.json")

    def __init__(self, db: Session):
        self.db = db
        self.service = None
        self._init_service()

    def _init_service(self):
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            creds = None
            if os.path.exists(self.TOKEN_PATH):
                creds = Credentials.from_authorized_user_file(self.TOKEN_PATH, self.SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif os.path.exists(self.CREDS_PATH):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.CREDS_PATH, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    with open(self.TOKEN_PATH, "w") as f:
                        f.write(creds.to_json())
                else:
                    print("[Calendar] credentials.json not found. "
                          "Place it at ~/.vector_ai/credentials.json to enable Google Calendar.")
                    return

            self.service = build("calendar", "v3", credentials=creds)
            print("[Calendar] Google Calendar connected.")
        except ImportError:
            print("[Calendar] Google API packages not installed. "
                  "Run: pip install google-api-python-client google-auth-oauthlib")
        except Exception as e:
            print(f"[Calendar] Init error: {e}")

    def is_available(self) -> bool:
        return self.service is not None

    def today(self) -> str:
        if not self.service:
            return "Google Calendar not connected. See setup instructions in tools.py."
        try:
            now = datetime.utcnow().isoformat() + "Z"
            end = (datetime.utcnow() + timedelta(days=1)).isoformat() + "Z"
            result = self.service.events().list(
                calendarId="primary",
                timeMin=now, timeMax=end,
                singleEvents=True, orderBy="startTime",
            ).execute()
            events = result.get("items", [])
            if not events:
                return "No events scheduled for today."
            lines = ["Today's Calendar:"]
            for e in events:
                start = e["start"].get("dateTime", e["start"].get("date", ""))
                try:
                    start_fmt = datetime.fromisoformat(start.replace("Z","")).strftime("%H:%M")
                except Exception:
                    start_fmt = start
                lines.append(f"  {start_fmt} — {e.get('summary', 'No title')}")
                if e.get("location"):
                    lines.append(f"    📍 {e['location']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Calendar fetch failed: {e}"

    def upcoming(self, days: int = 7) -> str:
        if not self.service:
            return "Google Calendar not connected."
        try:
            now = datetime.utcnow().isoformat() + "Z"
            end = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"
            result = self.service.events().list(
                calendarId="primary",
                timeMin=now, timeMax=end,
                singleEvents=True, orderBy="startTime",
                maxResults=10,
            ).execute()
            events = result.get("items", [])
            if not events:
                return f"No events in the next {days} days."
            lines = [f"Upcoming Events (next {days} days):"]
            for e in events:
                start = e["start"].get("dateTime", e["start"].get("date", ""))
                try:
                    dt = datetime.fromisoformat(start.replace("Z",""))
                    start_fmt = dt.strftime("%b %d, %H:%M")
                except Exception:
                    start_fmt = start
                lines.append(f"  {start_fmt} — {e.get('summary', 'No title')}")
            return "\n".join(lines)
        except Exception as e:
            return f"Calendar fetch failed: {e}"

    def add_event(self, title: str, start: datetime, end: datetime = None,
                  description: str = "", location: str = "") -> str:
        if not self.service:
            return "Google Calendar not connected."
        if not end:
            end = start + timedelta(hours=1)
        try:
            event = {
                "summary": title,
                "description": description,
                "location": location,
                "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Kolkata"},
                "end":   {"dateTime": end.isoformat(),   "timeZone": "Asia/Kolkata"},
            }
            created = self.service.events().insert(
                calendarId="primary", body=event
            ).execute()
            return f"Event created: '{title}' on {start.strftime('%b %d at %H:%M')}."
        except Exception as e:
            return f"Failed to create event: {e}"
