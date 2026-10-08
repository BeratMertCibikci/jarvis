"""Saves/loads the conversation history in SQLite.

Why a separate file? Because this isn't a "tool" given to the model, it's the server's own
infrastructure. The tools in tools.py are what the model calls; this is a store that
agent.py uses quietly before and after every message.

The two engines keep their history in different formats:
- Ollama: plain Python dictionaries (already JSON-serializable).
- Gemini: the SDK's own "Content" objects (need an extra step to become JSON).
That's why save/load take a "serialize/deserialize" function as a parameter; each engine
defines its own conversion in agent.py.
"""
import json
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

DB_PATH = os.getenv("JARVIS_DB", "jarvis.db")  # same file as tools.py
TZ = "Europe/Paris"

# Note that replaces old photos in the history. Re-sending a photo with every message costs
# a lot of tokens; the model already reads it and writes down what it needs the first time.
IMAGE_PLACEHOLDER = "[📷 photo]"


def split_recent(history: list, is_user_message, keep: int) -> tuple[list, list]:
    """Splits the history into (older, recent): the recent part starts with the last `keep`
    user messages.

    Why cut at a "user message" instead of after N messages? Because the history contains tool
    calls (model: "call this tool" -> tool: "here's the result"). Cutting in the middle of such
    a pair leaves the model with a call without a result, or a result without a call, and the
    API errors. A message the user typed is always a safe starting point.
    """
    starts = [i for i, m in enumerate(history) if is_user_message(m)]
    if len(starts) <= keep:
        return [], history
    cut = starts[-keep]
    return history[:cut], history[cut:]


def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS conversations (
        session_id TEXT NOT NULL,
        engine TEXT NOT NULL,
        history TEXT NOT NULL,
        updated TEXT NOT NULL,
        PRIMARY KEY (session_id, engine))""")
    # A tool call waiting for the user's approval (an approval card). At most one per chat.
    # Kept in the database so the card survives a page reload or a server restart.
    con.execute("""CREATE TABLE IF NOT EXISTS pending (
        session_id TEXT NOT NULL,
        engine TEXT NOT NULL,
        data TEXT NOT NULL,
        PRIMARY KEY (session_id, engine))""")
    return con


def load_pending(session_id: str, engine: str):
    with _db() as con:
        row = con.execute("SELECT data FROM pending WHERE session_id = ? AND engine = ?",
                          (session_id, engine)).fetchone()
    return json.loads(row[0]) if row else None


def save_pending(session_id: str, engine: str, data: dict) -> None:
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO pending (session_id, engine, data) VALUES (?, ?, ?)",
                    (session_id, engine, json.dumps(data, ensure_ascii=False, default=str)))


def clear_pending(session_id: str, engine: str) -> None:
    with _db() as con:
        con.execute("DELETE FROM pending WHERE session_id = ? AND engine = ?", (session_id, engine))


def load_history(session_id: str, engine: str, deserialize=lambda x: x) -> list:
    """Returns the saved history. If there is none, or it's corrupted, returns an empty list
    (the chat starts over instead of the app crashing)."""
    with _db() as con:
        row = con.execute(
            "SELECT history FROM conversations WHERE session_id = ? AND engine = ?",
            (session_id, engine),
        ).fetchone()
    if not row:
        return []
    try:
        return deserialize(json.loads(row[0]))
    except Exception as e:
        print(f"[memory] couldn't read the history, starting over: {e}")
        return []


def save_history(session_id: str, engine: str, history: list, serialize=lambda x: x) -> None:
    payload = json.dumps(serialize(history), ensure_ascii=False, default=str)
    now = datetime.now(ZoneInfo(TZ)).isoformat()
    with _db() as con:
        con.execute(
            """INSERT INTO conversations (session_id, engine, history, updated) VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id, engine) DO UPDATE SET history = excluded.history, updated = excluded.updated""",
            (session_id, engine, payload, now),
        )
