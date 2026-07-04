# VECTOR AI Mark 1
### Your Personal J.A.R.V.I.S — Built Right
## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FRONTEND (coming next)              │
│         React + Tailwind + Electron                  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (localhost:8000)
┌────────────────────▼────────────────────────────────┐
│              FASTAPI BACKEND                         │
│                                                      │
│   ┌─────────────┐    ┌──────────────────────────┐   │
│   │ Orchestrator│───▶│  Intent Detection        │   │
│   │  (brain)    │    │  (regex + LLM fallback)  │   │
│   └──────┬──────┘    └──────────────────────────┘   │
│          │                                           │
│    ┌─────▼──────────────────────────────────────┐   │
│    │               TOOLS                        │   │
│    │  Tasks | Reminders | News | Search         │   │
│    │  Progress | Google Calendar                │   │
│    └─────────────────────────────────────────────┘  │
│                                                      │
│   ┌──────────────┐    ┌───────────────────────────┐ │
│   │  Ollama LLM  │    │  SQLite Memory + DB       │ │
│   │  (local)     │    │  (semantic recall)        │ │
│   └──────────────┘    └───────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  VOICE LAYER                         │
│       Whisper STT + Edge TTS (en-GB-RyanNeural)      │
└─────────────────────────────────────────────────────┘
```

*"Good day. I am Vector. How may I assist you?"*
