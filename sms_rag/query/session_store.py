"""Session store for persisting chat history to SQLite.

Provides conversation continuity across browser sessions and server restarts.
Falls back to in-memory operation if SQLite is unavailable or fails.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from sms_rag.shared.models import SessionExchange

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    exchange_id TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    source_references TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
"""


class SessionStore:
    """Persists chat session history to a local SQLite database.

    Maintains up to MAX_EXCHANGES exchanges using FIFO eviction.
    Falls back to in-memory storage on SQLite failures.
    """

    MAX_EXCHANGES: int = 50

    def __init__(self, db_path: Path = Path("storage/sessions.db")) -> None:
        """Initialize the session store.

        Creates the database file and schema if they do not exist.
        Falls back to in-memory operation if SQLite is unavailable.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._in_memory: list[SessionExchange] = []
        self._using_memory = False

        try:
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Failed to initialize SQLite session store at %s: %s. "
                "Falling back to in-memory history.",
                db_path,
                exc,
            )
            self._conn = None
            self._using_memory = True

    def load_history(self) -> list[SessionExchange]:
        """Load all session exchanges ordered by timestamp ascending."""
        if self._using_memory:
            return list(self._in_memory)

        try:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT exchange_id, user_message, assistant_response, "
                "source_references, timestamp "
                "FROM sessions ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()
            return [
                SessionExchange(
                    exchange_id=row[0],
                    user_message=row[1],
                    assistant_response=row[2],
                    source_references=json.loads(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                )
                for row in rows
            ]
        except (sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to load session history from SQLite: %s. "
                "Falling back to in-memory history.",
                exc,
            )
            self._fallback_to_memory()
            return list(self._in_memory)

    def save_exchange(self, exchange: SessionExchange) -> None:
        """Persist a new exchange, evicting the oldest if at capacity."""
        if self._using_memory:
            self._in_memory.append(exchange)
            if len(self._in_memory) > self.MAX_EXCHANGES:
                self._in_memory.pop(0)
            return

        try:
            assert self._conn is not None
            # Check current count
            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]

            if count >= self.MAX_EXCHANGES:
                self._evict_oldest()

            self._conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(exchange_id, user_message, assistant_response, "
                "source_references, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    exchange.exchange_id,
                    exchange.user_message,
                    exchange.assistant_response,
                    json.dumps(exchange.source_references),
                    exchange.timestamp.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning(
                "Failed to save exchange to SQLite: %s. "
                "Falling back to in-memory history.",
                exc,
            )
            self._fallback_to_memory()
            self._in_memory.append(exchange)
            if len(self._in_memory) > self.MAX_EXCHANGES:
                self._in_memory.pop(0)

    def _evict_oldest(self) -> None:
        """Remove the oldest exchange to maintain the 50-exchange cap (FIFO)."""
        if self._using_memory:
            if self._in_memory:
                self._in_memory.pop(0)
            return

        try:
            assert self._conn is not None
            self._conn.execute(
                "DELETE FROM sessions WHERE exchange_id = ("
                "  SELECT exchange_id FROM sessions "
                "  ORDER BY timestamp ASC LIMIT 1"
                ")"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Failed to evict oldest exchange: %s", exc)

    def clear(self) -> None:
        """Delete all exchanges from the session history."""
        if self._using_memory:
            self._in_memory.clear()
            return

        try:
            assert self._conn is not None
            self._conn.execute("DELETE FROM sessions")
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning(
                "Failed to clear session history from SQLite: %s. "
                "Falling back to in-memory history.",
                exc,
            )
            self._fallback_to_memory()
            self._in_memory.clear()

    def _fallback_to_memory(self) -> None:
        """Switch to in-memory operation, preserving any loaded history."""
        if self._using_memory:
            return

        # Try to load existing history before switching
        history: list[SessionExchange] = []
        if self._conn is not None:
            try:
                cursor = self._conn.execute(
                    "SELECT exchange_id, user_message, assistant_response, "
                    "source_references, timestamp "
                    "FROM sessions ORDER BY timestamp ASC"
                )
                rows = cursor.fetchall()
                history = [
                    SessionExchange(
                        exchange_id=row[0],
                        user_message=row[1],
                        assistant_response=row[2],
                        source_references=json.loads(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                    )
                    for row in rows
                ]
            except (sqlite3.Error, json.JSONDecodeError, ValueError):
                pass

        self._using_memory = True
        self._in_memory = history
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
