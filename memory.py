"""
Vector AI — Memory Layer
Stores and retrieves long-term memories about the user.
Uses simple TF-IDF cosine similarity for semantic search
(no external embedding API needed — runs 100% locally).
"""

import json
import math
import re
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from database import MemoryChunk


# ── Simple local embeddings (TF-IDF bag-of-words) ────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def tf(tokens: List[str]) -> Dict[str, float]:
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens) or 1
    return {k: v / total for k, v in freq.items()}

def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    keys = set(vec_a) & set(vec_b)
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def embed(text: str) -> str:
    """Convert text to a TF vector, stored as JSON."""
    tokens = tokenize(text)
    vector = tf(tokens)
    return json.dumps(vector)

def similarity_from_stored(stored_embedding: str, query: str) -> float:
    """Compare a stored JSON embedding to a new query string."""
    try:
        vec_a = json.loads(stored_embedding)
        vec_b = tf(tokenize(query))
        return cosine_similarity(vec_a, vec_b)
    except Exception:
        return 0.0


# ── Memory Operations ─────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, db: Session):
        self.db = db

    def save(self, content: str, source: str = "conversation", importance: float = 1.0):
        """Save a new memory chunk."""
        chunk = MemoryChunk(
            content=content,
            source=source,
            embedding=embed(content),
            importance=importance,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk

    def recall(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """
        Retrieve the most relevant memories for a given query.
        Returns top_k results sorted by relevance × importance.
        """
        all_chunks = self.db.query(MemoryChunk).all()
        scored = []
        for chunk in all_chunks:
            sim = similarity_from_stored(chunk.embedding, query)
            score = sim * chunk.importance
            if score >= min_score:
                scored.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "score": round(score, 4),
                    "importance": chunk.importance,
                    "created_at": chunk.created_at,
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def extract_and_save(self, user_message: str, assistant_reply: str):
        """
        Automatically extract memorable facts from a conversation turn
        and save them. Simple heuristic — looks for personal facts,
        project names, preferences stated by the user.
        """
        triggers = [
            "my name is", "i am", "i'm", "i work", "i study",
            "i like", "i prefer", "i hate", "my project",
            "mckinsey", "college", "remind me", "always",
            "every day", "my goal", "i want to", "i need to",
        ]
        msg_lower = user_message.lower()
        if any(t in msg_lower for t in triggers):
            # Save the user message as a memory
            self.save(
                content=f"User said: {user_message}",
                source="conversation",
                importance=1.5,
            )

    def boost_importance(self, memory_id: int, amount: float = 0.5):
        """Increase importance of a memory (called when user references it again)."""
        chunk = self.db.query(MemoryChunk).filter(MemoryChunk.id == memory_id).first()
        if chunk:
            chunk.importance = min(chunk.importance + amount, 5.0)
            chunk.updated_at = datetime.utcnow()
            self.db.commit()

    def forget(self, memory_id: int):
        """Delete a specific memory."""
        chunk = self.db.query(MemoryChunk).filter(MemoryChunk.id == memory_id).first()
        if chunk:
            self.db.delete(chunk)
            self.db.commit()

    def summarize_for_context(self, query: str) -> str:
        """
        Returns a short paragraph of relevant memories to inject into LLM context.
        """
        memories = self.recall(query, top_k=4)
        if not memories:
            return ""
        lines = [m["content"] for m in memories]
        return "Relevant context from memory:\n" + "\n".join(f"- {l}" for l in lines)
