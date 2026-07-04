"""
Vector AI — Orchestrator
The real brain. Decides:
  - What the user wants (intent)
  - Which tool to call
  - What memory to retrieve
  - What context to give the LLM
  - What response to send back

This is the heart of Vector.
"""

import re
from datetime import datetime
from typing import Dict, Tuple
from sqlalchemy.orm import Session

from llm import llm, OllamaLLM
from memory import MemoryStore
from tools import TaskTool, ReminderTool, NewsTool, SearchTool, ProgressTool, CalendarTool
from database import Message


# ── Intent patterns — fast regex check before asking the LLM ─────────────────
INTENT_PATTERNS = {
    "add_task": [
        r"add task", r"create task", r"new task", r"remind me to do",
        r"add to.*list", r"task:", r"todo:"
    ],
    "list_tasks": [
        r"my tasks", r"list tasks", r"show tasks", r"what.*tasks",
        r"pending tasks", r"todo list"
    ],
    "complete_task": [
        r"complete task", r"done.*task", r"finish task", r"mark.*done",
        r"task.*done", r"completed task"
    ],
    "add_reminder": [
        r"remind me", r"set.*reminder", r"reminder.*at", r"alert me",
        r"don't let me forget"
    ],
    "list_reminders": [
        r"my reminders", r"show reminders", r"upcoming reminders",
        r"what.*reminders"
    ],
    "log_progress": [
        r"log.*progress", r"i (did|completed|finished|worked on)",
        r"progress.*:", r"update.*progress", r"log.*mckinsey",
        r"i (studied|read|watched|attended)"
    ],
    "show_progress": [
        r"my progress", r"show progress", r"progress.*summary",
        r"what.*progress", r"progress.*report"
    ],
    "get_news": [
        r"news", r"headlines", r"what.*happening", r"latest.*news",
        r"news.*about", r"top stories"
    ],
    "search_web": [
        r"search.*for", r"look up", r"find.*information", r"what is ",
        r"who is ", r"search ", r"google "
    ],
    "calendar_today": [
        r"today.*calendar", r"my schedule", r"what.*today",
        r"calendar.*today", r"meetings today", r"events today"
    ],
    "calendar_upcoming": [
        r"upcoming.*events", r"next.*week.*calendar", r"schedule.*week",
        r"what.*week", r"events.*coming"
    ],
    "calendar_add": [
        r"add.*calendar", r"create.*event", r"schedule.*meeting",
        r"book.*meeting", r"add.*meeting"
    ],
    "system_status": [
        r"status", r"system status", r"diagnostics", r"are you online",
        r"how are you", r"vector status"
    ],
    "memory_recall": [
        r"do you remember", r"what.*know.*about me", r"recall",
        r"my.*information", r"what did i tell you"
    ],
}


def detect_intent(text: str) -> str:
    """Fast regex-based intent detection. Falls back to LLM if unsure."""
    lower = text.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                return intent
    return "chat"  # default — send to LLM


class Orchestrator:
    def __init__(self, db: Session, news_api_key: str = ""):
        self.db = db
        self.memory    = MemoryStore(db)
        self.tasks     = TaskTool(db)
        self.reminders = ReminderTool(db)
        self.news      = NewsTool(api_key=news_api_key)
        self.search    = SearchTool()
        self.progress  = ProgressTool(db)
        self.calendar  = CalendarTool(db)

    def _save_message(self, role: str, content: str, session_id: str):
        msg = Message(role=role, content=content, session_id=session_id,
                      timestamp=datetime.utcnow())
        self.db.add(msg)
        self.db.commit()

    def _get_history(self, session_id: str, limit: int = 10):
        messages = (
            self.db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        messages.reverse()
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_context(self, user_input: str) -> str:
        """Gather context from DB and memory to inject into LLM."""
        pending = self.tasks.list_pending()
        projects = self.progress.active_projects()
        memories = self.memory.recall(user_input, top_k=3)

        return llm.generate_system_context({
            "pending_tasks": pending,
            "active_projects": projects,
            "memories": memories,
        })

    def process(self, user_input: str, session_id: str = "default") -> Dict:
        """
        Main entry point.
        Returns: {reply, intent, tool_used}
        """
        # 1. Detect intent
        intent = detect_intent(user_input)

        # 2. If intent is ambiguous, ask the LLM to classify
        if intent == "chat" and llm.is_online():
            llm_intent = llm.classify_intent(user_input)
            if llm_intent != "chat":
                intent = llm_intent

        # 3. Save user message to history
        self._save_message("user", user_input, session_id)

        # 4. Route to appropriate tool or LLM
        reply, tool_used = self._route(intent, user_input, session_id)

        # 5. Save assistant reply to history
        self._save_message("assistant", reply, session_id)

        # 6. Extract and save memories from this turn
        self.memory.extract_and_save(user_input, reply)

        return {"reply": reply, "intent": intent, "tool_used": tool_used}

    def _route(self, intent: str, user_input: str, session_id: str) -> Tuple[str, str]:
        """Route to the right tool based on intent."""
        lower = user_input.lower()

        # ── Tasks ─────────────────────────────────────────────────────────────
        if intent == "add_task":
            reply = self.tasks.parse_and_add(user_input)
            return reply, "task_tool"

        if intent == "list_tasks":
            reply = self.tasks.format_list()
            return reply, "task_tool"

        if intent == "complete_task":
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                reply = self.tasks.complete(int(numbers[0]))
            else:
                reply = "Please specify which task number to complete. E.g. 'complete task 3'"
            return reply, "task_tool"

        # ── Reminders ─────────────────────────────────────────────────────────
        if intent == "add_reminder":
            remind_at = self.reminders.parse_time(user_input)
            title = re.sub(
                r"(remind me|to|at|tomorrow|tonight|in \d+ (hour|minute)s?)",
                "", lower
            ).strip().title() or user_input
            reply = self.reminders.add(title=title, remind_at=remind_at)
            return reply, "reminder_tool"

        if intent == "list_reminders":
            reply = self.reminders.list_upcoming()
            return reply, "reminder_tool"

        # ── Progress ──────────────────────────────────────────────────────────
        if intent == "log_progress":
            project = "General"
            for kw in ["mckinsey", "college", "work", "personal", "forward", "semester"]:
                if kw in lower:
                    project = kw.title()
                    break
            entry = user_input
            reply = self.progress.log(project=project, entry=entry)
            return reply, "progress_tool"

        if intent == "show_progress":
            project = None
            for kw in ["mckinsey", "college", "work", "personal", "forward"]:
                if kw in lower:
                    project = kw.title()
                    break
            reply = self.progress.summary(project=project)
            return reply, "progress_tool"

        # ── News ──────────────────────────────────────────────────────────────
        if intent == "get_news":
            query = re.sub(r"(news|headlines|about|latest|top|stories)", "", lower).strip()
            if query:
                reply = self.news.search(query)
            else:
                reply = self.news.top_headlines()
            return reply, "news_tool"

        # ── Web Search ────────────────────────────────────────────────────────
        if intent == "search_web":
            query = re.sub(r"(search|for|look up|find|google)", "", lower).strip()
            reply = self.search.search(query or user_input)
            return reply, "search_tool"

        # ── Calendar ──────────────────────────────────────────────────────────
        if intent == "calendar_today":
            reply = self.calendar.today()
            return reply, "calendar_tool"

        if intent == "calendar_upcoming":
            reply = self.calendar.upcoming()
            return reply, "calendar_tool"

        if intent == "calendar_add":
            # Let LLM extract event details then add
            context = self._build_context(user_input)
            history = self._get_history(session_id)
            extract_prompt = (
                f"Extract from this request: title, date/time, location. "
                f"Request: '{user_input}'. Reply with: TITLE | DATETIME | LOCATION"
            )
            extracted = llm.chat([{"role": "user", "content": extract_prompt}])
            parts = extracted.split("|")
            if len(parts) >= 2:
                title = parts[0].strip()
                try:
                    from dateutil import parser as dateparser
                    dt = dateparser.parse(parts[1].strip())
                except Exception:
                    dt = datetime.now()
                location = parts[2].strip() if len(parts) > 2 else ""
                reply = self.calendar.add_event(title, dt, location=location)
            else:
                reply = "I couldn't extract event details. Please say: 'Add event: [title] on [date] at [time]'"
            return reply, "calendar_tool"

        # ── System Status ─────────────────────────────────────────────────────
        if intent == "system_status":
            pending = len(self.tasks.list_pending())
            projects = self.progress.active_projects()
            reply = "\n".join([
                "╔══ VECTOR — SYSTEM STATUS ══╗",
                f"  Time     : {datetime.now().strftime('%H:%M — %A %b %d, %Y')}",
                f"  LLM      : {'Ollama Online ✓' if llm.is_online() else 'Ollama Offline ✗'}",
                f"  Calendar : {'Connected ✓' if self.calendar.is_available() else 'Not configured'}",
                f"  Tasks    : {pending} pending",
                f"  Projects : {', '.join(projects) if projects else 'None yet'}",
                "╚════════════════════════════╝",
            ])
            return reply, "system"

        # ── Memory Recall ─────────────────────────────────────────────────────
        if intent == "memory_recall":
            memories = self.memory.recall(user_input, top_k=5)
            if not memories:
                return "I don't have specific memories stored about that yet.", "memory"
            lines = ["Here's what I remember:"]
            for m in memories:
                lines.append(f"  • {m['content']}")
            return "\n".join(lines), "memory"

        # ── Default: Chat with LLM ────────────────────────────────────────────
        context = self._build_context(user_input)
        history = self._get_history(session_id)
        reply = llm.chat(messages=history, system_extra=context)
        return reply, "llm"
