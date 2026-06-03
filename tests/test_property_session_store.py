"""Property-based tests for the session store.

# Feature: sms-rag, Property 18: Session Persistence Round-Trip
# Feature: sms-rag, Property 19: Session FIFO Eviction
# Feature: sms-rag, Property 20: Session Store Graceful Degradation

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.6
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sms_rag.query.session_store import SessionStore
from sms_rag.shared.models import SessionExchange


# --- Generators ---


@st.composite
def source_reference_generator(draw):
    """Generate a single source reference dict with string keys and values."""
    keys = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P"),
                    blacklist_characters=("\x00",),
                ),
                min_size=1,
                max_size=20,
            ),
            min_size=0,
            max_size=5,
            unique=True,
        )
    )
    ref = {}
    for key in keys:
        ref[key] = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                    blacklist_characters=("\x00",),
                ),
                min_size=0,
                max_size=50,
            )
        )
    return ref


@st.composite
def session_exchange_generator(draw):
    """Generate random SessionExchange objects with valid fields.

    Produces exchanges with:
    - exchange_id: non-empty alphanumeric string
    - user_message: non-empty text string
    - assistant_response: non-empty text string
    - source_references: list of dicts with string keys/values
    - timestamp: a datetime within a reasonable range
    """
    exchange_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=36,
        )
    )
    user_message = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_characters=("\x00",),
            ),
            min_size=1,
            max_size=200,
        )
    )
    assistant_response = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_characters=("\x00",),
            ),
            min_size=1,
            max_size=200,
        )
    )
    source_references = draw(
        st.lists(source_reference_generator(), min_size=0, max_size=5)
    )
    # Use timestamps in a reasonable range that round-trip through ISO format
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31),
        )
    )
    return SessionExchange(
        exchange_id=exchange_id,
        user_message=user_message,
        assistant_response=assistant_response,
        source_references=source_references,
        timestamp=timestamp,
    )


@st.composite
def unique_session_exchanges_generator(draw, min_size=1, max_size=10):
    """Generate a list of SessionExchange objects with unique exchange_ids and timestamps."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    exchanges = []
    used_ids = set()

    base_time = datetime(2024, 1, 1)

    for i in range(count):
        ex = draw(session_exchange_generator())
        # Ensure unique exchange_id by appending index
        ex.exchange_id = f"{ex.exchange_id}-{i}"
        if ex.exchange_id in used_ids:
            ex.exchange_id = f"uniq-{i}"
        used_ids.add(ex.exchange_id)
        # Assign monotonically increasing timestamps to guarantee ordering
        ex.timestamp = base_time + timedelta(seconds=i)
        exchanges.append(ex)

    return exchanges


# --- Property 18: Session Persistence Round-Trip ---


class TestSessionPersistenceRoundTrip:
    """Property 18: Session Persistence Round-Trip.

    For any valid session exchange, saving and loading SHALL return
    identical values.

    # Feature: sms-rag, Property 18: Session Persistence Round-Trip
    **Validates: Requirements 10.1, 10.2, 10.3**
    """

    @given(exchange=session_exchange_generator())
    @settings(max_examples=25)
    def test_single_exchange_round_trip(self, exchange: SessionExchange):
        """For any valid session exchange, saving and loading SHALL return
        an exchange with identical user_message, assistant_response,
        source_references, and timestamp values.

        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_round_trip.db"
            store = SessionStore(db_path=db_path)

            store.save_exchange(exchange)
            history = store.load_history()

            assert len(history) == 1
            loaded = history[0]

            assert loaded.exchange_id == exchange.exchange_id
            assert loaded.user_message == exchange.user_message
            assert loaded.assistant_response == exchange.assistant_response
            assert loaded.source_references == exchange.source_references
            assert loaded.timestamp == exchange.timestamp

    @given(exchanges=unique_session_exchanges_generator(min_size=1, max_size=10))
    @settings(max_examples=25)
    def test_multiple_exchanges_round_trip(self, exchanges: list[SessionExchange]):
        """For any list of valid session exchanges, saving all and loading
        SHALL return exchanges with identical values in timestamp order.

        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_multi_round_trip.db"
            store = SessionStore(db_path=db_path)

            for ex in exchanges:
                store.save_exchange(ex)

            history = store.load_history()

            assert len(history) == len(exchanges)

            # Exchanges should be in timestamp order
            sorted_exchanges = sorted(exchanges, key=lambda e: e.timestamp)
            for loaded, original in zip(history, sorted_exchanges):
                assert loaded.exchange_id == original.exchange_id
                assert loaded.user_message == original.user_message
                assert loaded.assistant_response == original.assistant_response
                assert loaded.source_references == original.source_references
                assert loaded.timestamp == original.timestamp

    @given(exchange=session_exchange_generator())
    @settings(max_examples=25)
    def test_persistence_across_instances(self, exchange: SessionExchange):
        """For any valid session exchange, saving in one store instance and
        loading in another SHALL return identical values.

        **Validates: Requirements 10.1, 10.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_persistence.db"

            store1 = SessionStore(db_path=db_path)
            store1.save_exchange(exchange)

            store2 = SessionStore(db_path=db_path)
            history = store2.load_history()

            assert len(history) == 1
            loaded = history[0]
            assert loaded.exchange_id == exchange.exchange_id
            assert loaded.user_message == exchange.user_message
            assert loaded.assistant_response == exchange.assistant_response
            assert loaded.source_references == exchange.source_references
            assert loaded.timestamp == exchange.timestamp


# --- Property 19: Session FIFO Eviction ---


class TestSessionFIFOEviction:
    """Property 19: Session FIFO Eviction.

    For any session with N >= 50 exchanges, adding one more SHALL result
    in exactly 50, the new one present, oldest removed.

    # Feature: sms-rag, Property 19: Session FIFO Eviction
    **Validates: Requirements 10.4**
    """

    @given(
        new_exchange=session_exchange_generator(),
        n_extra=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=25, deadline=None)
    def test_fifo_eviction_at_capacity(
        self, new_exchange: SessionExchange, n_extra: int
    ):
        """For any session with N >= 50 exchanges, adding one more SHALL
        result in exactly 50 stored exchanges, the newly added exchange
        SHALL be present, and the exchange with the oldest timestamp SHALL
        no longer be present.

        **Validates: Requirements 10.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_eviction.db"
            store = SessionStore(db_path=db_path)

            # Fill to capacity (50) plus n_extra
            total_prefill = 50 + n_extra
            base_time = datetime(2024, 1, 1)
            for i in range(total_prefill):
                ex = SessionExchange(
                    exchange_id=f"prefill-{i}",
                    user_message=f"msg-{i}",
                    assistant_response=f"resp-{i}",
                    source_references=[],
                    timestamp=base_time + timedelta(seconds=i),
                )
                store.save_exchange(ex)

            # Get the current state before adding the new exchange
            history_before = store.load_history()
            assert len(history_before) == 50

            oldest_before = history_before[0]

            # Ensure new exchange has a unique ID and a newer timestamp
            new_exchange.exchange_id = "new-exchange-unique"
            new_exchange.timestamp = base_time + timedelta(seconds=total_prefill + 1)

            store.save_exchange(new_exchange)

            history_after = store.load_history()

            # Exactly 50 exchanges
            assert len(history_after) == 50

            # New exchange is present
            ids_after = [h.exchange_id for h in history_after]
            assert new_exchange.exchange_id in ids_after

            # Oldest from before is removed
            assert oldest_before.exchange_id not in ids_after

    @given(num_over=st.integers(min_value=1, max_value=20))
    @settings(max_examples=25, deadline=None)
    def test_eviction_maintains_cap_after_multiple_additions(self, num_over: int):
        """For any number of additions beyond capacity, the store SHALL
        maintain exactly 50 exchanges.

        **Validates: Requirements 10.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_cap.db"
            store = SessionStore(db_path=db_path)

            total = 50 + num_over
            base_time = datetime(2024, 1, 1)
            for i in range(total):
                ex = SessionExchange(
                    exchange_id=f"ex-{i}",
                    user_message=f"msg-{i}",
                    assistant_response=f"resp-{i}",
                    source_references=[],
                    timestamp=base_time + timedelta(seconds=i),
                )
                store.save_exchange(ex)

            history = store.load_history()
            assert len(history) == 50

            # The newest exchanges should be present
            ids = [h.exchange_id for h in history]
            for i in range(num_over, total):
                assert f"ex-{i}" in ids

            # The oldest evicted exchanges should NOT be present
            for i in range(num_over):
                assert f"ex-{i}" not in ids


# --- Property 20: Session Store Graceful Degradation ---


class TestSessionStoreGracefulDegradation:
    """Property 20: Session Store Graceful Degradation.

    For any sequence of operations after SQLite failure, the store SHALL
    not raise exceptions and SHALL fall back to in-memory.

    # Feature: sms-rag, Property 20: Session Store Graceful Degradation
    **Validates: Requirements 10.6**
    """

    @given(exchanges=unique_session_exchanges_generator(min_size=1, max_size=20))
    @settings(max_examples=25)
    def test_corrupted_db_falls_back_to_memory(self, exchanges: list[SessionExchange]):
        """For any sequence of operations after SQLite failure (corrupted DB),
        the store SHALL not raise exceptions and SHALL fall back to in-memory.

        **Validates: Requirements 10.6**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "corrupted.db"
            # Write garbage to simulate corruption
            db_path.write_bytes(b"CORRUPTED DATA " * 100)

            # Should not raise - falls back to in-memory
            store = SessionStore(db_path=db_path)

            # Operations should work without raising exceptions
            for ex in exchanges:
                store.save_exchange(ex)

            history = store.load_history()
            assert len(history) == min(len(exchanges), SessionStore.MAX_EXCHANGES)

            # All saved exchanges should be retrievable
            saved_ids = {h.exchange_id for h in history}
            expected_ids = {
                ex.exchange_id for ex in exchanges[-SessionStore.MAX_EXCHANGES :]
            }
            assert saved_ids == expected_ids

    @given(exchanges=unique_session_exchanges_generator(min_size=1, max_size=20))
    @settings(max_examples=25)
    def test_permission_denied_falls_back_to_memory(
        self, exchanges: list[SessionExchange]
    ):
        """For any sequence of operations after permission-denied failure,
        the store SHALL not raise exceptions and SHALL fall back to in-memory.

        **Validates: Requirements 10.6**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a path where we can't create the db file
            # (use a file as parent so mkdir fails)
            blocker = Path(tmp_dir) / "blocker_file"
            blocker.write_text("I am a file blocking directory creation")
            db_path = blocker / "sessions.db"

            # Should not raise - falls back to in-memory
            store = SessionStore(db_path=db_path)

            # Operations should work without exceptions
            for ex in exchanges:
                store.save_exchange(ex)

            history = store.load_history()
            assert len(history) == min(len(exchanges), SessionStore.MAX_EXCHANGES)

    @given(
        initial_exchanges=unique_session_exchanges_generator(min_size=1, max_size=10),
        post_failure_exchanges=unique_session_exchanges_generator(
            min_size=1, max_size=10
        ),
    )
    @settings(max_examples=25)
    def test_in_memory_fallback_operations_functional(
        self,
        initial_exchanges: list[SessionExchange],
        post_failure_exchanges: list[SessionExchange],
    ):
        """For any sequence of save and load operations in fallback mode,
        subsequent operations SHALL function correctly using the in-memory store.

        **Validates: Requirements 10.6**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Force in-memory mode via invalid path
            blocker = Path(tmp_dir) / "blocker"
            blocker.write_text("blocking")
            db_path = blocker / "sessions.db"

            store = SessionStore(db_path=db_path)

            # Save initial exchanges
            for ex in initial_exchanges:
                store.save_exchange(ex)

            # Verify they're loadable
            history = store.load_history()
            assert len(history) == len(initial_exchanges)

            # Save more exchanges (ensure unique IDs from initial ones)
            for i, ex in enumerate(post_failure_exchanges):
                ex.exchange_id = f"post-{i}-{ex.exchange_id}"
                ex.timestamp = datetime(2025, 1, 1) + timedelta(seconds=i)
                store.save_exchange(ex)

            # Verify all are loadable
            history = store.load_history()
            total_expected = min(
                len(initial_exchanges) + len(post_failure_exchanges),
                SessionStore.MAX_EXCHANGES,
            )
            assert len(history) == total_expected

            # Clear should work too
            store.clear()
            assert store.load_history() == []

    @given(exchanges=unique_session_exchanges_generator(min_size=1, max_size=10))
    @settings(max_examples=25)
    def test_in_memory_fifo_eviction_works(self, exchanges: list[SessionExchange]):
        """For any session in fallback mode that reaches capacity,
        FIFO eviction SHALL work correctly.

        **Validates: Requirements 10.6**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Force in-memory mode
            blocker = Path(tmp_dir) / "blocker"
            blocker.write_text("blocking")
            db_path = blocker / "sessions.db"

            store = SessionStore(db_path=db_path)

            # Fill to capacity
            base_time = datetime(2024, 1, 1)
            for i in range(50):
                ex = SessionExchange(
                    exchange_id=f"fill-{i}",
                    user_message=f"msg-{i}",
                    assistant_response=f"resp-{i}",
                    source_references=[],
                    timestamp=base_time + timedelta(seconds=i),
                )
                store.save_exchange(ex)

            # Add the test exchanges beyond capacity
            for i, ex in enumerate(exchanges):
                ex.exchange_id = f"overflow-{i}"
                ex.timestamp = base_time + timedelta(seconds=50 + i)
                store.save_exchange(ex)

            history = store.load_history()
            assert len(history) == 50

            # The overflow exchanges should be present
            ids = [h.exchange_id for h in history]
            for i in range(len(exchanges)):
                assert f"overflow-{i}" in ids
