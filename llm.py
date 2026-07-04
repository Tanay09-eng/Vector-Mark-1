"""
Vector AI — LLM Layer
Talks to Ollama running locally.
Model: llama3 (default) — swap to mistral, gemma, phi3 etc.

Install Ollama: https://ollama.com
Pull model:     ollama pull llama3
Run server:     ollama serve  (auto-starts on install)
"""

import requests
import json
from typing import List, Dict, Generator

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3"

VECTOR_SYSTEM = """You are Vector — a highly sophisticated personal AI assistant, modeled after J.A.R.V.I.S from Iron Man. You serve your user with unwavering professionalism, precision, and calm intelligence.

Personality:
- Formal, composed, and precise — never casual
- Dry wit when appropriate
- Address the user respectfully, like a trusted advisor
- Proactive — anticipate needs and suggest next steps

You have access to: tasks, reminders, progress tracking, news, Google Calendar, and web search.
When the user asks you to do something actionable (add task, set reminder, log progress, get news, check calendar), respond with a clear confirmation of what you did.
Keep responses concise unless depth is requested.
Current context will be injected by the orchestrator."""


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        self.model = model
        self.base_url = base_url

    def is_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def available_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(
        self,
        messages: List[Dict],
        system_extra: str = "",
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Send a list of {role, content} messages to Ollama.
        Returns the assistant reply as a string.
        """
        system = VECTOR_SYSTEM
        if system_extra:
            system += f"\n\n{system_extra}"

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return (
                "Ollama is not running. Please start it with: ollama serve\n"
                "Then make sure you have pulled a model: ollama pull llama3"
            )
        except Exception as e:
            return f"LLM error: {e}"

    def classify_intent(self, user_input: str) -> str:
        """
        Ask the LLM to classify the user's intent into one of the
        orchestrator's known tool categories.
        Returns a short intent string.
        """
        prompt = f"""Classify this user message into exactly ONE of these intents:
chat, add_task, list_tasks, complete_task, add_reminder, list_reminders,
log_progress, show_progress, get_news, search_web, calendar_today,
calendar_add, system_status, memory_recall

User message: "{user_input}"

Reply with ONLY the intent word, nothing else."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 20},
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=15,
            )
            r.raise_for_status()
            intent = r.json()["message"]["content"].strip().lower().split()[0]
            valid = {
                "chat", "add_task", "list_tasks", "complete_task",
                "add_reminder", "list_reminders", "log_progress",
                "show_progress", "get_news", "search_web",
                "calendar_today", "calendar_add", "system_status",
                "memory_recall",
            }
            return intent if intent in valid else "chat"
        except Exception:
            return "chat"

    def generate_system_context(self, context_data: dict) -> str:
        """Build a context string injected into every LLM call."""
        parts = []
        from datetime import datetime
        parts.append(f"Current time: {datetime.now().strftime('%A, %B %d %Y — %H:%M')}")

        if context_data.get("pending_tasks"):
            tasks = context_data["pending_tasks"]
            parts.append(f"Pending tasks ({len(tasks)}): " +
                         ", ".join(t["title"] for t in tasks[:5]))

        if context_data.get("upcoming_reminders"):
            reminders = context_data["upcoming_reminders"]
            parts.append(f"Upcoming reminders: " +
                         ", ".join(r["title"] for r in reminders[:3]))

        if context_data.get("active_projects"):
            parts.append(f"Active projects: {', '.join(context_data['active_projects'])}")

        if context_data.get("memories"):
            mem_text = " | ".join(m["content"] for m in context_data["memories"][:3])
            parts.append(f"What I know about you: {mem_text}")

        return "\n".join(parts)


# Singleton
llm = OllamaLLM()
