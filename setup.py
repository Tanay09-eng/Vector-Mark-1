"""
Vector AI v2 — First Time Setup
Run this ONCE before starting Vector.

What it does:
1. Creates ~/.vector_ai/ config folder
2. Asks for your API keys and saves them
3. Checks if Ollama is installed
4. Tells you exactly what to install if anything is missing
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASE_DIR = Path.home() / ".vector_ai"
BASE_DIR.mkdir(exist_ok=True)
CONFIG_FILE = BASE_DIR / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def check_ollama():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║            VECTOR AI v2 — SETUP WIZARD                  ║
╚══════════════════════════════════════════════════════════╝
""")

def main():
    print_banner()
    cfg = load_config()

    # ── Step 1: NewsAPI Key ───────────────────────────────────────────────────
    print("STEP 1 — NewsAPI Key")
    print("  (You already have this from your old NewsApp project)")
    existing = cfg.get("news_api_key", "")
    if existing:
        print(f"  ✓ Already saved: {existing[:8]}...")
        change = input("  Change it? (y/N): ").strip().lower()
        if change == "y":
            key = input("  Paste NewsAPI key: ").strip()
            if key:
                cfg["news_api_key"] = key
    else:
        key = input("  Paste your NewsAPI key (or press Enter to skip): ").strip()
        if key:
            cfg["news_api_key"] = key

    # ── Step 2: Ollama ────────────────────────────────────────────────────────
    print("\nSTEP 2 — Ollama (Local LLM)")
    if check_ollama():
        print("  ✓ Ollama is running!")
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            if models:
                print(f"  ✓ Available models: {', '.join(models)}")
                cfg["ollama_model"] = models[0]
            else:
                print("  ⚠ No models pulled yet.")
                print("  Run this in a NEW terminal: ollama pull llama3")
        except Exception:
            pass
    else:
        print("  ✗ Ollama is NOT running or not installed.")
        print()
        print("  To install Ollama:")
        print("  1. Go to: https://ollama.com")
        print("  2. Download and install for Windows")
        print("  3. Open a terminal and run: ollama pull llama3")
        print("  4. Ollama will run automatically in the background")
        print()
        input("  Press Enter once you've installed Ollama to continue...")

    # ── Step 3: Google Calendar ───────────────────────────────────────────────
    print("\nSTEP 3 — Google Calendar (Optional)")
    print("  To enable Google Calendar:")
    print("  1. Go to: https://console.cloud.google.com/")
    print("  2. Create a project → Enable 'Google Calendar API'")
    print("  3. Create OAuth2 credentials (Desktop App)")
    print("  4. Download credentials.json")
    print(f"  5. Place it here: {BASE_DIR / 'credentials.json'}")
    print()
    creds_path = BASE_DIR / "credentials.json"
    if creds_path.exists():
        print("  ✓ credentials.json found! Google Calendar will work.")
    else:
        print("  ℹ  credentials.json not found. Calendar disabled for now.")
        print("     You can add it later — Vector works without it.")

    # ── Save config ───────────────────────────────────────────────────────────
    save_config(cfg)
    print(f"\n  ✓ Config saved to {CONFIG_FILE}")

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  To start Vector, run these TWO commands:               ║
║                                                          ║
║  Terminal 1 (keep open):                                ║
║    ollama serve                                          ║
║                                                          ║
║  Terminal 2:                                            ║
║    cd vector_v2/backend                                  ║
║    uvicorn main:app --reload --port 8000                ║
║                                                          ║
║  Then open: http://localhost:8000/docs                   ║
║  (Full UI coming next — the React/Electron frontend)    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
