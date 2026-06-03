"""Property-based tests for RAG orchestrator (Properties 11, 12).

# Feature: sms-rag, Property 11: Orchestrator Context Chunk Cap
# Feature: sms-rag, Property 12: Orchestrator Output Completeness

Validates: Requirements 6.5, 6.7
"""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sms_rag.query.orchestrator import RAGOrchestrator
from sms_rag.shared.models import GenerationResult, SearchResult


# --- Generators ---


@st.composite
def search_result_generator(draw):
    """Generate a random SearchResult with realistic metadata."""
    chunk_id = draw(
        st.text(
            min_size=3,
            max_size=15,
            alphabet=st.characters(
                whitelist_categories=("L", "N"), blacklist_characters="\x00"
            ),
        ).map(lambda s: f"chunk_{s}")
    )
    text = draw(
        st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"), blacklist_characters="\x00"
            ),
        )
    )
    score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    participant = draw(st.sampled_from(["Alice", "Bob", "Charlie", "Diana", "Eve"]))
    source = draw(st.sampled_from(["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"]))
    metadata = {
        "participant_name": participant,
        "source_filename": source,
        "date_range_start": "2024-01-01",
        "date_range_end": "2024-01-15",
    }
    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, score=score)


@st.composite
def chunk_list_generator(draw, min_count=5, max_count=25):
    """Generate a list of SearchResult chunks with a count between min and max."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    chunks = draw(st.lists(search_result_generator(), min_size=count, max_size=count))
    return chunks


@st.composite
def query_generator(draw):
    """Generate random query strings with at least one word."""
    words = draw(
        st.lists(
            st.sampled_from(
                [
                    "hello",
                    "what",
                    "when",
                    "who",
                    "message",
                    "conversation",
                    "phone",
                    "said",
                    "time",
                    "about",
                    "tell",
                    "find",
                    "chat",
                    "call",
                ]
            ),
            min_size=1,
            max_size=5,
        )
    )
    return " ".join(words)


@st.composite
def max_context_chunks_generator(draw):
    """Generate max_context_chunks values between 1 and 20."""
    return draw(st.integers(min_value=1, max_value=20))


@st.composite
def llm_response_generator(draw):
    """Generate non-empty LLM response strings."""
    response = draw(
        st.text(
            min_size=10,
            max_size=200,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z", "P"),
                blacklist_characters="\x00",
            ),
        )
    )
    # Ensure non-empty after stripping
    assume(response.strip())
    return response


# --- Helpers ---


def _create_mock_retriever(chunks_to_return: list[SearchResult]):
    """Create a mock retriever that returns the given chunks."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = chunks_to_return
    return mock_retriever


def _create_mock_llm_provider(response_text: str):
    """Create a mock LLM provider that returns the given text."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = response_text
    return mock_provider


# --- Property 11: Orchestrator Context Chunk Cap ---


class TestOrchestratorContextChunkCap:
    """Property 11: Orchestrator Context Chunk Cap.

    # Feature: sms-rag, Property 11: Orchestrator Context Chunk Cap

    For any query that retrieves N chunks (where N may exceed 10), the
    LLM_Orchestrator SHALL pass at most 10 chunks as context to the
    LLM_Provider.
    """

    @given(
        query=query_generator(),
        num_chunks=st.integers(min_value=5, max_value=25),
        max_context_chunks=st.just(10),
    )
    @settings(max_examples=100)
    def test_context_chunks_capped_at_default(
        self, query, num_chunks, max_context_chunks
    ):
        """With default max_context_chunks=10, at most 10 chunks SHALL be
        passed as context regardless of how many the retriever returns.

        **Validates: Requirements 6.5**
        """
        # Generate chunks that the retriever will return
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"passage content number {i} with some text",
                metadata={
                    "participant_name": "Alice",
                    "source_filename": "file1.pdf",
                    "date_range_start": "2024-01-01",
                    "date_range_end": "2024-01-15",
                },
                score=1.0 - (i * 0.03),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider("A generated response about the query.")

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=max_context_chunks,
        )

        result = orchestrator.query(query)

        # The source_chunks in the result should be capped
        assert len(result.source_chunks) <= max_context_chunks, (
            f"Got {len(result.source_chunks)} source_chunks, "
            f"expected at most {max_context_chunks}"
        )

    @given(
        query=query_generator(),
        num_chunks=st.integers(min_value=5, max_value=25),
        max_context_chunks=max_context_chunks_generator(),
    )
    @settings(max_examples=100)
    def test_context_chunks_capped_at_configurable_limit(
        self, query, num_chunks, max_context_chunks
    ):
        """For any configurable max_context_chunks value, the orchestrator
        SHALL pass at most that many chunks as context.

        **Validates: Requirements 6.5**
        """
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"passage content number {i}",
                metadata={
                    "participant_name": "Bob",
                    "source_filename": "file2.pdf",
                    "date_range_start": "2024-02-01",
                    "date_range_end": "2024-02-28",
                },
                score=1.0 - (i * 0.02),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider("Response text from the LLM.")

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=max_context_chunks,
        )

        result = orchestrator.query(query)

        # source_chunks should never exceed max_context_chunks
        assert len(result.source_chunks) <= max_context_chunks, (
            f"Got {len(result.source_chunks)} source_chunks, "
            f"expected at most {max_context_chunks}. "
            f"Retriever returned {num_chunks} chunks."
        )

    @given(
        query=query_generator(),
        num_chunks=st.integers(min_value=11, max_value=25),
    )
    @settings(max_examples=100)
    def test_context_exceeding_cap_is_truncated(self, query, num_chunks):
        """When retriever returns more than 10 chunks, exactly 10 SHALL
        be passed to the LLM (not more).

        **Validates: Requirements 6.5**
        """
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"message text for chunk {i}",
                metadata={
                    "participant_name": "Charlie",
                    "source_filename": "file3.pdf",
                    "date_range_start": "2024-03-01",
                    "date_range_end": "2024-03-31",
                },
                score=1.0 - (i * 0.01),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider("Generated answer.")

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=10,
        )

        result = orchestrator.query(query)

        # Since retriever returned > 10, result should be exactly 10
        assert len(result.source_chunks) == 10, (
            f"Got {len(result.source_chunks)} source_chunks when retriever "
            f"returned {num_chunks} chunks, expected exactly 10"
        )


# --- Property 12: Orchestrator Output Completeness ---


class TestOrchestratorOutputCompleteness:
    """Property 12: Orchestrator Output Completeness.

    # Feature: sms-rag, Property 12: Orchestrator Output Completeness

    For any successful LLM generation, the GenerationResult SHALL contain
    both a non-empty text response and a non-empty list of source chunk
    references.
    """

    @given(
        query=query_generator(),
        llm_response=llm_response_generator(),
        num_chunks=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=100)
    def test_generation_result_has_non_empty_text(
        self, query, llm_response, num_chunks
    ):
        """GenerationResult SHALL contain a non-empty text response.

        **Validates: Requirements 6.7**
        """
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"relevant passage {i}",
                metadata={
                    "participant_name": "Alice",
                    "source_filename": "file1.pdf",
                    "date_range_start": "2024-01-01",
                    "date_range_end": "2024-01-15",
                },
                score=0.9 - (i * 0.05),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider(llm_response)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=10,
        )

        result = orchestrator.query(query)

        assert result.text, (
            f"GenerationResult.text is empty or None. "
            f"LLM was expected to return: '{llm_response}'"
        )
        assert len(result.text) > 0, "GenerationResult.text has zero length"

    @given(
        query=query_generator(),
        llm_response=llm_response_generator(),
        num_chunks=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=100)
    def test_generation_result_has_non_empty_source_chunks(
        self, query, llm_response, num_chunks
    ):
        """GenerationResult SHALL contain a non-empty list of source chunk
        references when the retriever returns results.

        **Validates: Requirements 6.7**
        """
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"source passage {i} with content",
                metadata={
                    "participant_name": "Diana",
                    "source_filename": "file4.pdf",
                    "date_range_start": "2024-04-01",
                    "date_range_end": "2024-04-30",
                },
                score=0.95 - (i * 0.05),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider(llm_response)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=10,
        )

        result = orchestrator.query(query)

        assert result.source_chunks, (
            "GenerationResult.source_chunks is empty. "
            "Expected non-empty list of source chunk references."
        )
        assert (
            len(result.source_chunks) > 0
        ), "GenerationResult.source_chunks has zero length"

    @given(
        query=query_generator(),
        llm_response=llm_response_generator(),
        num_chunks=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=100)
    def test_generation_result_completeness_combined(
        self, query, llm_response, num_chunks
    ):
        """GenerationResult SHALL have BOTH non-empty text AND non-empty
        source_chunks simultaneously for any successful generation.

        **Validates: Requirements 6.7**
        """
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"conversation excerpt {i}",
                metadata={
                    "participant_name": "Eve",
                    "source_filename": "file5.pdf",
                    "date_range_start": "2024-05-01",
                    "date_range_end": "2024-05-31",
                },
                score=0.85 - (i * 0.04),
            )
            for i in range(num_chunks)
        ]

        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider(llm_response)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=10,
        )

        result = orchestrator.query(query)

        # Both conditions must hold simultaneously
        assert result.text and result.source_chunks, (
            f"GenerationResult incomplete: "
            f"text={'non-empty' if result.text else 'EMPTY'}, "
            f"source_chunks={len(result.source_chunks) if result.source_chunks else 'EMPTY'}"
        )
