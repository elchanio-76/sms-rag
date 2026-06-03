"""Unit tests for the ChatInterface class."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sms_rag.query.chat_ui import (
    ChatInterface,
    _EMPTY_INPUT_MSG,
    _EMPTY_STORE_BANNER,
    _TIMEOUT_ERROR_MSG,
    _format_source_references,
)
from sms_rag.shared.models import GenerationResult, SearchResult, SessionExchange


# --- Fixtures ---


@pytest.fixture
def mock_orchestrator():
    """Create a mock RAGOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.query.return_value = GenerationResult(
        text="This is a test response.",
        source_chunks=[
            SearchResult(
                chunk_id="chunk_1",
                text="Some text",
                metadata={
                    "participant_name": "Alice",
                    "date_range_start": "2024-01-01",
                    "date_range_end": "2024-01-15",
                },
                score=0.9,
            )
        ],
    )
    return orchestrator


@pytest.fixture
def mock_session_store():
    """Create a mock SessionStore."""
    store = MagicMock()
    store.load_history.return_value = []
    return store


@pytest.fixture
def chat_interface(mock_orchestrator, mock_session_store):
    """Create a ChatInterface instance with mocks."""
    return ChatInterface(
        orchestrator=mock_orchestrator,
        session_store=mock_session_store,
    )


# --- Tests for _format_source_references ---


class TestFormatSourceReferences:
    def test_empty_sources_returns_empty_string(self):
        result = _format_source_references([])
        assert result == ""

    def test_single_source_with_full_metadata(self):
        chunk = SearchResult(
            chunk_id="c1",
            text="text",
            metadata={
                "participant_name": "Bob",
                "date_range_start": "2024-03-01",
                "date_range_end": "2024-03-15",
            },
            score=0.8,
        )
        result = _format_source_references([chunk])
        assert "<details>" in result
        assert "Sources" in result
        assert "Bob" in result
        assert "2024-03-01" in result
        assert "2024-03-15" in result

    def test_source_with_only_start_date(self):
        chunk = SearchResult(
            chunk_id="c2",
            text="text",
            metadata={
                "participant_name": "Carol",
                "date_range_start": "2024-05-01",
            },
            score=0.7,
        )
        result = _format_source_references([chunk])
        assert "Carol" in result
        assert "2024-05-01" in result

    def test_source_with_no_dates(self):
        chunk = SearchResult(
            chunk_id="c3",
            text="text",
            metadata={"participant_name": "Dave"},
            score=0.6,
        )
        result = _format_source_references([chunk])
        assert "Dave" in result
        assert "–" not in result

    def test_source_with_missing_participant(self):
        chunk = SearchResult(
            chunk_id="c4",
            text="text",
            metadata={},
            score=0.5,
        )
        result = _format_source_references([chunk])
        assert "Unknown" in result

    def test_multiple_sources(self):
        chunks = [
            SearchResult(
                chunk_id=f"c{i}",
                text="text",
                metadata={"participant_name": f"Person{i}"},
                score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]
        result = _format_source_references(chunks)
        assert "Person0" in result
        assert "Person1" in result
        assert "Person2" in result


# --- Tests for ChatInterface._respond ---


class TestChatInterfaceRespond:
    def test_empty_message_rejected(self, chat_interface):
        response = chat_interface._respond("", [])
        assert response == _EMPTY_INPUT_MSG

    def test_whitespace_only_message_rejected(self, chat_interface):
        response = chat_interface._respond("   \t\n  ", [])
        assert response == _EMPTY_INPUT_MSG

    def test_valid_message_returns_response_with_sources(
        self, chat_interface, mock_orchestrator
    ):
        response = chat_interface._respond("Hello", [])
        assert "This is a test response." in response
        assert "<details>" in response
        assert "Alice" in response
        mock_orchestrator.query.assert_called_once_with("Hello")

    def test_message_with_leading_trailing_whitespace_stripped(
        self, chat_interface, mock_orchestrator
    ):
        chat_interface._respond("  Hello  ", [])
        mock_orchestrator.query.assert_called_once_with("Hello")

    def test_timeout_returns_error_message(self, chat_interface, mock_orchestrator):
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        mock_orchestrator.query.side_effect = lambda q: (_ for _ in ()).throw(
            FuturesTimeoutError()
        )
        # Need to simulate actual timeout by making the future timeout
        with patch("sms_rag.query.chat_ui.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__ = MagicMock(
                return_value=mock_executor
            )
            mock_executor_class.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.side_effect = FuturesTimeoutError()
            mock_executor.submit.return_value = mock_future

            response = chat_interface._respond("slow query", [])
            assert response == _TIMEOUT_ERROR_MSG

    def test_orchestrator_error_returns_error_message(
        self, chat_interface, mock_orchestrator
    ):
        with patch("sms_rag.query.chat_ui.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__ = MagicMock(
                return_value=mock_executor
            )
            mock_executor_class.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.side_effect = RuntimeError("LLM unavailable")
            mock_executor.submit.return_value = mock_future

            response = chat_interface._respond("test query", [])
            assert "error occurred" in response.lower() or "❌" in response

    def test_exchange_persisted_to_session_store(
        self, chat_interface, mock_session_store
    ):
        chat_interface._respond("test message", [])
        mock_session_store.save_exchange.assert_called_once()
        exchange = mock_session_store.save_exchange.call_args[0][0]
        assert exchange.user_message == "test message"
        assert "This is a test response." in exchange.assistant_response


# --- Tests for session history loading ---


class TestSessionHistoryLoading:
    def test_loads_history_from_session_store(self, mock_orchestrator):
        store = MagicMock()
        store.load_history.return_value = [
            SessionExchange(
                exchange_id="ex1",
                user_message="Hi",
                assistant_response="Hello!",
                source_references=[],
                timestamp=datetime(2024, 1, 1, 12, 0),
            )
        ]
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=store)
        history = interface._load_session_history()
        assert len(history) == 2  # user + assistant messages
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello!"}

    def test_falls_back_to_in_memory_on_store_failure(self, mock_orchestrator):
        store = MagicMock()
        store.load_history.side_effect = RuntimeError("DB error")
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=store)
        # After init failure, should be using in-memory
        assert interface._session_store_available is False

    def test_no_session_store_uses_in_memory(self, mock_orchestrator):
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)
        assert interface._session_store_available is False
        history = interface._load_session_history()
        assert history == []


# --- Tests for session store fallback ---


class TestSessionStoreFallback:
    def test_fallback_to_memory_on_save_failure(self, mock_orchestrator):
        store = MagicMock()
        store.load_history.return_value = []
        store.save_exchange.side_effect = RuntimeError("Write error")

        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=store)
        interface._respond("test", [])

        # Should have fallen back to in-memory
        assert interface._session_store_available is False
        assert len(interface._in_memory_history) == 1

    def test_in_memory_history_capped_at_50(self, mock_orchestrator):
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        for i in range(55):
            interface._in_memory_history.append(
                SessionExchange(
                    exchange_id=str(i),
                    user_message=f"msg {i}",
                    assistant_response=f"resp {i}",
                    source_references=[],
                    timestamp=datetime.now(),
                )
            )

        # Simulate persist which enforces the cap
        interface._persist_exchange("new msg", "new resp", [])
        assert len(interface._in_memory_history) <= 50


# --- Tests for vector store empty detection ---


class TestVectorStoreEmptyCheck:
    def test_detects_empty_store_via_collection(self):
        # Use a non-Mock object to avoid MagicMock auto-creating attributes
        class FakeOrchestrator:
            pass

        orchestrator = FakeOrchestrator()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_store = MagicMock()
        mock_store._collection = mock_collection
        mock_retriever = MagicMock()
        mock_retriever._vector_store = mock_store
        orchestrator._retriever = mock_retriever

        interface = ChatInterface(orchestrator=orchestrator, session_store=None)
        assert interface._check_vector_store_empty() is True

    def test_detects_non_empty_store(self):
        class FakeOrchestrator:
            pass

        orchestrator = FakeOrchestrator()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_store = MagicMock()
        mock_store._collection = mock_collection
        mock_retriever = MagicMock()
        mock_retriever._vector_store = mock_store
        orchestrator._retriever = mock_retriever

        interface = ChatInterface(orchestrator=orchestrator, session_store=None)
        assert interface._check_vector_store_empty() is False

    def test_uses_is_store_empty_method_if_available(self):
        class FakeOrchestrator:
            def is_store_empty(self):
                return True

        orchestrator = FakeOrchestrator()
        interface = ChatInterface(orchestrator=orchestrator, session_store=None)
        assert interface._check_vector_store_empty() is True

    def test_returns_true_on_exception(self):
        class FakeOrchestrator:
            def is_store_empty(self):
                raise RuntimeError("fail")

        orchestrator = FakeOrchestrator()
        interface = ChatInterface(orchestrator=orchestrator, session_store=None)
        assert interface._check_vector_store_empty() is True
