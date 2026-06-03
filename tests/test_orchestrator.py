"""Unit tests for the RAG orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from sms_rag.query.orchestrator import RAGOrchestrator
from sms_rag.shared.models import GenerationResult, SearchResult


def _make_search_result(
    chunk_id: str = "chunk_1",
    text: str = "Hello, how are you?",
    participant: str = "John Doe",
    date_start: str = "2024-01-01",
    date_end: str = "2024-01-15",
    score: float = 0.9,
) -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "participant_name": participant,
            "date_range_start": date_start,
            "date_range_end": date_end,
        },
        score=score,
    )


class TestRAGOrchestrator:
    """Tests for RAGOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_retriever = MagicMock()
        self.mock_llm_provider = MagicMock()
        self.orchestrator = RAGOrchestrator(
            retriever=self.mock_retriever,
            llm_provider=self.mock_llm_provider,
            max_context_chunks=10,
        )

    def test_query_returns_generation_result(self):
        """query() should return a GenerationResult with text and sources."""
        chunks = [_make_search_result()]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_llm_provider.generate.return_value = "The answer is 42."

        result = self.orchestrator.query("What is the meaning of life?")

        assert isinstance(result, GenerationResult)
        assert result.text == "The answer is 42."
        assert result.source_chunks == chunks

    def test_query_caps_context_chunks(self):
        """query() should cap context to max_context_chunks."""
        # Create 15 chunks but cap is 10
        chunks = [
            _make_search_result(chunk_id=f"chunk_{i}", score=1.0 - i * 0.05)
            for i in range(15)
        ]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_llm_provider.generate.return_value = "Response"

        orchestrator = RAGOrchestrator(
            retriever=self.mock_retriever,
            llm_provider=self.mock_llm_provider,
            max_context_chunks=10,
        )
        result = orchestrator.query("test query")

        # Should only include first 10 chunks as source references
        assert len(result.source_chunks) == 10
        assert result.source_chunks[0].chunk_id == "chunk_0"
        assert result.source_chunks[9].chunk_id == "chunk_9"

    def test_query_passes_metadata_filters_to_retriever(self):
        """query() should pass metadata_filters to the retriever."""
        self.mock_retriever.retrieve.return_value = []
        self.mock_llm_provider.generate.return_value = "No results."

        filters = {"participant_name": "Alice"}
        self.orchestrator.query("hello", metadata_filters=filters)

        self.mock_retriever.retrieve.assert_called_once_with(
            query="hello",
            metadata_filters=filters,
            top_k=10,
        )

    def test_query_with_no_results(self):
        """query() should handle empty retrieval gracefully."""
        self.mock_retriever.retrieve.return_value = []
        self.mock_llm_provider.generate.return_value = "I couldn't find anything."

        result = self.orchestrator.query("something obscure")

        assert result.text == "I couldn't find anything."
        assert result.source_chunks == []

    def test_query_calls_llm_with_context(self):
        """query() should pass context texts to the LLM provider."""
        chunks = [_make_search_result(text="Hi there!")]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_llm_provider.generate.return_value = "Got it."

        self.orchestrator.query("greet me")

        # LLM provider should have been called with prompt and context
        self.mock_llm_provider.generate.assert_called_once()
        call_args = self.mock_llm_provider.generate.call_args
        assert call_args.kwargs["prompt"] == "greet me"
        # Context should be a list of formatted message strings
        assert isinstance(call_args.kwargs["context"], list)
        assert len(call_args.kwargs["context"]) > 0

    def test_query_with_custom_max_context(self):
        """Constructor should accept custom max_context_chunks."""
        chunks = [_make_search_result(chunk_id=f"c_{i}") for i in range(5)]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_llm_provider.generate.return_value = "Response"

        orchestrator = RAGOrchestrator(
            retriever=self.mock_retriever,
            llm_provider=self.mock_llm_provider,
            max_context_chunks=3,
        )
        result = orchestrator.query("test")

        assert len(result.source_chunks) == 3

    def test_format_context_includes_source_info(self):
        """_format_context should include participant name and date range."""
        chunk = _make_search_result(
            participant="Maria", date_start="2024-03-01", date_end="2024-03-10"
        )
        context = self.orchestrator._format_context([chunk])

        assert "Maria" in context
        assert "2024-03-01" in context
        assert "2024-03-10" in context
        assert "[Passage 1]" in context

    def test_format_context_with_partial_dates(self):
        """_format_context should handle missing date_start or date_end."""
        chunk_no_end = SearchResult(
            chunk_id="c1",
            text="text",
            metadata={
                "participant_name": "Bob",
                "date_range_start": "2024-01-01",
                "date_range_end": "",
            },
            score=0.8,
        )
        context = self.orchestrator._format_context([chunk_no_end])
        assert "Bob" in context
        assert "from 2024-01-01" in context

    def test_format_context_no_dates(self):
        """_format_context should fall back to participant name only."""
        chunk = SearchResult(
            chunk_id="c1",
            text="text",
            metadata={
                "participant_name": "Charlie",
                "date_range_start": "",
                "date_range_end": "",
            },
            score=0.7,
        )
        context = self.orchestrator._format_context([chunk])
        assert "Charlie" in context

    def test_format_context_empty_chunks(self):
        """_format_context with empty chunks returns a fallback message."""
        context = self.orchestrator._format_context([])
        assert "No relevant" in context

    def test_llm_provider_error_propagates(self):
        """RuntimeError from LLM provider should propagate."""
        self.mock_retriever.retrieve.return_value = [_make_search_result()]
        self.mock_llm_provider.generate.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            self.orchestrator.query("test")
