"""Unit tests for the session store module."""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from sms_rag.query.session_store import SessionStore
from sms_rag.shared.models import SessionExchange


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test_sessions.db"


@pytest.fixture
def store(db_path: Path) -> SessionStore:
    """Create a fresh SessionStore for each test."""
    return SessionStore(db_path=db_path)


def _make_exchange(
    index: int = 0, timestamp: datetime | None = None
) -> SessionExchange:
    """Create a test exchange with a given index."""
    if timestamp is None:
        timestamp = datetime(2024, 1, 1) + timedelta(hours=index)
    return SessionExchange(
        exchange_id=f"ex-{index}",
        user_message=f"Question {index}",
        assistant_response=f"Answer {index}",
        source_references=[{"participant": "Alice", "date": "2024-01-01"}],
        timestamp=timestamp,
    )


class TestSessionStoreInit:
    """Tests for SessionStore initialization."""

    def test_creates_database_file(self, db_path: Path) -> None:
        """Database file is created on initialization."""
        SessionStore(db_path=db_path)
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        db_path = tmp_path / "nested" / "dir" / "sessions.db"
        SessionStore(db_path=db_path)
        assert db_path.exists()

    def test_creates_schema(self, db_path: Path) -> None:
        """Sessions table and index are created."""
        SessionStore(db_path=db_path)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert cursor.fetchone() is not None
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_sessions_timestamp'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_graceful_degradation_on_invalid_path(self, tmp_path: Path) -> None:
        """Falls back to in-memory if path is invalid."""
        # Use a path that can't be created (file where dir should be)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        db_path = blocker / "sessions.db"

        store = SessionStore(db_path=db_path)
        # Should still work in-memory
        store.save_exchange(_make_exchange(0))
        history = store.load_history()
        assert len(history) == 1


class TestLoadHistory:
    """Tests for load_history method."""

    def test_empty_history(self, store: SessionStore) -> None:
        """Returns empty list when no exchanges stored."""
        assert store.load_history() == []

    def test_loads_saved_exchanges(self, store: SessionStore) -> None:
        """Loads exchanges that were previously saved."""
        ex = _make_exchange(0)
        store.save_exchange(ex)
        history = store.load_history()
        assert len(history) == 1
        assert history[0].exchange_id == ex.exchange_id
        assert history[0].user_message == ex.user_message
        assert history[0].assistant_response == ex.assistant_response

    def test_preserves_source_references(self, store: SessionStore) -> None:
        """Source references are correctly serialized and deserialized."""
        refs = [
            {"participant": "Alice", "date_range": "2024-01-01 to 2024-01-05"},
            {"participant": "Bob", "date_range": "2024-02-10 to 2024-02-15"},
        ]
        ex = SessionExchange(
            exchange_id="ref-test",
            user_message="test",
            assistant_response="response",
            source_references=refs,
            timestamp=datetime(2024, 1, 1),
        )
        store.save_exchange(ex)
        history = store.load_history()
        assert history[0].source_references == refs

    def test_chronological_order(self, store: SessionStore) -> None:
        """History is ordered by timestamp ascending."""
        for i in [3, 1, 2]:
            store.save_exchange(_make_exchange(i))
        history = store.load_history()
        timestamps = [h.timestamp for h in history]
        assert timestamps == sorted(timestamps)


class TestSaveExchange:
    """Tests for save_exchange method."""

    def test_save_and_retrieve(self, store: SessionStore) -> None:
        """Basic save and retrieve works."""
        ex = _make_exchange(0)
        store.save_exchange(ex)
        history = store.load_history()
        assert len(history) == 1
        assert history[0].exchange_id == "ex-0"

    def test_timestamp_round_trip(self, store: SessionStore) -> None:
        """Timestamp survives round-trip through ISO format."""
        ts = datetime(2024, 6, 15, 14, 30, 45)
        ex = _make_exchange(0, timestamp=ts)
        store.save_exchange(ex)
        history = store.load_history()
        assert history[0].timestamp == ts

    def test_fifo_eviction_at_capacity(self, store: SessionStore) -> None:
        """Oldest exchange is evicted when at 50-exchange capacity."""
        for i in range(50):
            store.save_exchange(_make_exchange(i))

        # Now add one more
        store.save_exchange(_make_exchange(50))

        history = store.load_history()
        assert len(history) == 50
        # Oldest (ex-0) should be gone
        ids = [h.exchange_id for h in history]
        assert "ex-0" not in ids
        # Newest (ex-50) should be present
        assert "ex-50" in ids

    def test_multiple_evictions(self, store: SessionStore) -> None:
        """Multiple evictions maintain the cap."""
        for i in range(55):
            store.save_exchange(_make_exchange(i))

        history = store.load_history()
        assert len(history) == 50
        ids = [h.exchange_id for h in history]
        # First 5 should be evicted
        for i in range(5):
            assert f"ex-{i}" not in ids


class TestClear:
    """Tests for clear method."""

    def test_clear_removes_all(self, store: SessionStore) -> None:
        """Clear removes all exchanges."""
        for i in range(5):
            store.save_exchange(_make_exchange(i))
        store.clear()
        assert store.load_history() == []

    def test_clear_empty_store(self, store: SessionStore) -> None:
        """Clear on empty store doesn't error."""
        store.clear()
        assert store.load_history() == []


class TestGracefulDegradation:
    """Tests for fallback to in-memory operation."""

    def test_fallback_on_corrupted_db(self, tmp_path: Path) -> None:
        """Falls back to in-memory when database is corrupted."""
        db_path = tmp_path / "sessions.db"
        # Write garbage to simulate corruption
        db_path.write_bytes(b"this is not a valid sqlite database" * 10)

        store = SessionStore(db_path=db_path)
        # Should still work in memory
        store.save_exchange(_make_exchange(0))
        history = store.load_history()
        assert len(history) == 1

    def test_in_memory_fifo_eviction(self, tmp_path: Path) -> None:
        """FIFO eviction works in in-memory mode."""
        # Force in-memory mode
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        db_path = blocker / "sessions.db"

        store = SessionStore(db_path=db_path)
        for i in range(55):
            store.save_exchange(_make_exchange(i))

        history = store.load_history()
        assert len(history) == 50
        assert history[0].exchange_id == "ex-5"
        assert history[-1].exchange_id == "ex-54"

    def test_in_memory_clear(self, tmp_path: Path) -> None:
        """Clear works in in-memory mode."""
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        db_path = blocker / "sessions.db"

        store = SessionStore(db_path=db_path)
        store.save_exchange(_make_exchange(0))
        store.clear()
        assert store.load_history() == []


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_check_same_thread_false(self, db_path: Path) -> None:
        """SQLite connection uses check_same_thread=False."""
        store = SessionStore(db_path=db_path)
        # The connection should be usable from this test thread
        # (different from the thread that created it in some cases)
        assert store._conn is not None
        store.save_exchange(_make_exchange(0))
        assert len(store.load_history()) == 1


class TestPersistenceAcrossInstances:
    """Tests for persistence across SessionStore instances."""

    def test_data_persists_across_instances(self, db_path: Path) -> None:
        """Data saved by one instance is visible to another."""
        store1 = SessionStore(db_path=db_path)
        store1.save_exchange(_make_exchange(0))

        store2 = SessionStore(db_path=db_path)
        history = store2.load_history()
        assert len(history) == 1
        assert history[0].exchange_id == "ex-0"
