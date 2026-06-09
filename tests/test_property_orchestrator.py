"""Property-based tests for RAG orchestrator (Properties 1, 11, 12).

# Feature: sms-rag, Property 11: Orchestrator Context Chunk Cap
# Feature: sms-rag, Property 12: Orchestrator Output Completeness
# Feature: participant-context-clarity, Property 1: Participant Grouping Completeness
# Feature: participant-context-clarity, Property 2: Participant Section Header Format

Validates: Requirements 1.1, 1.6, 6.5, 6.7, 1.2
"""

import re
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sms_rag.query.orchestrator import RAGOrchestrator, _SYSTEM_TEMPLATE
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


# --- Generators for participant-context-clarity properties ---


@st.composite
def participant_name_generator(draw):
    """Generate realistic participant name strings.

    Produces names with various patterns: simple ASCII names, names with
    spaces, names with accented/unicode characters, and multi-word names.
    """
    # Mix of strategies covering different name patterns
    name = draw(
        st.one_of(
            # Simple ASCII names
            st.sampled_from(
                [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "Diana",
                    "Eve",
                    "John Smith",
                    "Mary Jane Watson",
                ]
            ),
            # Names with unicode/accented characters (Greek, accented Latin)
            st.sampled_from(
                [
                    "Kyriaki Salavanitou",
                    "Loizos Markides",
                    "José García",
                    "André Müller",
                    "François Leclerc",
                    "Ελένη Παπαδοπούλου",
                    "Νίκος Καζαντζάκης",
                ]
            ),
            # Generated text names (at least 1 char, printable, no newlines)
            st.text(
                min_size=1,
                max_size=40,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "Zs"),
                    blacklist_characters="\x00\n\r",
                ),
            ).filter(lambda s: s.strip()),
        )
    )
    return name


# --- Property 2: Participant Section Header Format ---


class TestParticipantSectionHeaderFormat:
    """Property 2: Participant Section Header Format.

    # Feature: participant-context-clarity, Property 2: Participant Section Header Format

    For any participant name string, the Context_Formatter SHALL produce
    a section header matching exactly the pattern
    `## Conversation with {name} (Your private conversation)`.

    **Validates: Requirements 1.2**
    """

    @given(name=participant_name_generator())
    @settings(max_examples=100)
    def test_section_header_matches_expected_format(self, name):
        """For any participant name, the header SHALL match exactly
        `## Conversation with {name} (Your private conversation)`.

        **Validates: Requirements 1.2**
        """
        # Create a single SearchResult with the generated participant name
        chunk = SearchResult(
            chunk_id="chunk_1",
            text="Some message text content here.",
            metadata={
                "participant_name": name,
                "source_filename": "test.pdf",
                "date_range_start": "2024-01-01",
                "date_range_end": "2024-01-15",
            },
            score=0.9,
        )

        # Use the orchestrator's _format_context directly
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )
        formatted = orchestrator._format_context([chunk])

        # The expected header
        expected_header = f"## Conversation with {name} (Your private conversation)"

        # Assert the header appears in the output
        assert expected_header in formatted, (
            f"Expected header '{expected_header}' not found in formatted output.\n"
            f"Got:\n{formatted}"
        )

    @given(name=participant_name_generator())
    @settings(max_examples=100)
    def test_section_header_is_first_line_of_section(self, name):
        """The participant section header SHALL be the first line of
        the section for that participant.

        **Validates: Requirements 1.2**
        """
        chunk = SearchResult(
            chunk_id="chunk_1",
            text="Hello world message.",
            metadata={
                "participant_name": name,
                "source_filename": "test.pdf",
            },
            score=0.85,
        )

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )
        formatted = orchestrator._format_context([chunk])

        expected_header = f"## Conversation with {name} (Your private conversation)"

        # The formatted output should start with the header
        # (since there's only one participant, it should be the very first line)
        first_line = formatted.split("\n")[0]
        assert first_line == expected_header, (
            f"First line of formatted output should be the header.\n"
            f"Expected: '{expected_header}'\n"
            f"Got: '{first_line}'"
        )

    @given(
        names=st.lists(
            participant_name_generator(),
            min_size=2,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_each_participant_gets_correct_header(self, names):
        """For multiple participants, each SHALL have a header matching
        the expected format with their specific name.

        **Validates: Requirements 1.2**
        """
        # Create one chunk per participant
        chunks = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"Message from participant {i}.",
                metadata={
                    "participant_name": name,
                    "source_filename": "test.pdf",
                    "date_range_start": "2024-01-01",
                    "date_range_end": "2024-01-15",
                },
                score=0.9 - (i * 0.1),
            )
            for i, name in enumerate(names)
        ]

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )
        formatted = orchestrator._format_context(chunks)

        # Each participant name should have its correctly-formatted header
        for name in names:
            expected_header = f"## Conversation with {name} (Your private conversation)"
            assert expected_header in formatted, (
                f"Expected header for '{name}' not found in output.\n"
                f"Expected: '{expected_header}'\n"
                f"Output:\n{formatted}"
            )


import re


# --- Generators for participant-context-clarity properties ---


@st.composite
def multi_participant_search_results_generator(draw):
    """Generate SearchResult lists from multiple participants with distinct scores.

    Ensures at least 2 distinct participants with varying max scores so we
    can verify section ordering.
    """
    # Choose 2-5 participant names
    all_participants = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    num_participants = draw(st.integers(min_value=2, max_value=5))
    participants = draw(
        st.lists(
            st.sampled_from(all_participants),
            min_size=num_participants,
            max_size=num_participants,
            unique=True,
        )
    )

    results = []
    for participant in participants:
        # Each participant gets 1-4 passages
        num_passages = draw(st.integers(min_value=1, max_value=4))
        for i in range(num_passages):
            score = draw(
                st.floats(
                    min_value=0.01,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            text = draw(
                st.text(
                    min_size=5,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("L", "N", "Z"),
                        blacklist_characters="\x00",
                    ),
                )
            )
            results.append(
                SearchResult(
                    chunk_id=f"chunk_{participant}_{i}",
                    text=text,
                    metadata={
                        "participant_name": participant,
                        "source_filename": f"{participant.lower()}.pdf",
                        "date_range_start": f"2024-01-{(i + 1):02d}",
                        "date_range_end": f"2024-01-{(i + 2):02d}",
                    },
                    score=score,
                )
            )

    return results


# --- Property 5: Participant Section Ordering by Score ---


class TestParticipantSectionOrderingByScore:
    """Property 5: Participant Section Ordering by Score.

    # Feature: participant-context-clarity, Property 5: Participant Section Ordering by Score

    For any set of SearchResults from multiple participants with varying scores,
    participant sections in the formatted output SHALL be ordered such that the
    section whose passages have the highest max score appears first, and each
    subsequent section has a max score less than or equal to the preceding section.

    **Validates: Requirements 1.8**
    """

    @given(chunks=multi_participant_search_results_generator())
    @settings(max_examples=100)
    def test_sections_ordered_by_descending_max_score(self, chunks):
        """Participant sections SHALL appear in descending order of their
        highest passage score.

        **Validates: Requirements 1.8**
        """
        # Build the orchestrator with a mock retriever (not used directly)
        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider("Response.")

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=50,
        )

        # Call _format_context directly to test formatting logic
        formatted = orchestrator._format_context(chunks)

        # Extract section headers in order from the formatted output
        header_pattern = r"## Conversation with (.+?) \(Your private conversation\)"
        section_participants = re.findall(header_pattern, formatted)

        # Compute the expected order: participants sorted by max score descending
        groups: dict[str, list[float]] = {}
        for chunk in chunks:
            participant = chunk.metadata.get("participant_name", "Unknown")
            groups.setdefault(participant, []).append(chunk.score)

        expected_order = sorted(
            groups.keys(),
            key=lambda p: max(groups[p]),
            reverse=True,
        )

        assert section_participants == expected_order, (
            f"Sections not ordered by descending max score.\n"
            f"Got order: {section_participants}\n"
            f"Expected order: {expected_order}\n"
            f"Max scores: {[(p, max(groups[p])) for p in expected_order]}"
        )

    @given(chunks=multi_participant_search_results_generator())
    @settings(max_examples=100)
    def test_each_subsequent_section_has_lower_or_equal_max_score(self, chunks):
        """Each subsequent participant section SHALL have a max score less than
        or equal to the preceding section's max score.

        **Validates: Requirements 1.8**
        """
        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider("Response.")

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=50,
        )

        formatted = orchestrator._format_context(chunks)

        # Extract section headers in order
        header_pattern = r"## Conversation with (.+?) \(Your private conversation\)"
        section_participants = re.findall(header_pattern, formatted)

        # Compute max scores per participant
        groups: dict[str, list[float]] = {}
        for chunk in chunks:
            participant = chunk.metadata.get("participant_name", "Unknown")
            groups.setdefault(participant, []).append(chunk.score)

        # Verify monotonically non-increasing max scores
        max_scores = [max(groups[p]) for p in section_participants]
        for i in range(len(max_scores) - 1):
            assert max_scores[i] >= max_scores[i + 1], (
                f"Section ordering violation at position {i}: "
                f"participant '{section_participants[i]}' has max score {max_scores[i]} "
                f"but is followed by '{section_participants[i + 1]}' with max score "
                f"{max_scores[i + 1]} (should be <= preceding)"
            )


# --- Property 1: Participant Grouping Completeness ---
# Feature: participant-context-clarity, Property 1: Participant Grouping Completeness


@st.composite
def search_result_with_participant_generator(draw, participant_names=None):
    """Generate a SearchResult with a configurable participant_name.

    Args:
        participant_names: Optional list of participant names to sample from.
            Defaults to a diverse set of names.
    """
    if participant_names is None:
        participant_names = [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Eve",
            "Kyriaki",
            "Loizos",
        ]

    chunk_id = draw(
        st.text(
            min_size=3,
            max_size=10,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ).map(lambda s: f"chunk_{s}")
    )
    # Generate unique text that won't collide between passages
    text = draw(
        st.text(
            min_size=10,
            max_size=80,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00\n",
            ),
        )
    )
    assume(text.strip())

    score = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    participant = draw(st.sampled_from(participant_names))

    # Optionally include date metadata
    has_dates = draw(st.booleans())
    metadata: dict = {"participant_name": participant, "source_filename": "test.pdf"}
    if has_dates:
        metadata["date_range_start"] = "2024-01-01"
        metadata["date_range_end"] = "2024-01-15"

    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, score=score)


class TestParticipantGroupingCompleteness:
    """Property 1: Participant Grouping Completeness.

    # Feature: participant-context-clarity, Property 1: Participant Grouping Completeness

    For any list of SearchResult objects with varying participant_name metadata
    values, the Context_Formatter SHALL produce output where every input passage
    appears exactly once, grouped under the header for its participant name, with
    no passages discarded or duplicated.

    **Validates: Requirements 1.1, 1.6**
    """

    @given(
        chunks=st.lists(
            search_result_with_participant_generator(),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=100)
    def test_every_passage_appears_exactly_once(self, chunks: list[SearchResult]):
        """Every input passage text SHALL appear exactly once in formatted output.

        **Validates: Requirements 1.1, 1.6**
        """
        # Ensure we have non-empty, distinct passage texts
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        output = orchestrator._format_context(chunks)

        # Every passage text must appear exactly once in the output
        for chunk in chunks:
            count = output.count(chunk.text)
            assert count == 1, (
                f"Passage '{chunk.text[:50]}...' appears {count} times in output, "
                f"expected exactly 1. Participant: {chunk.metadata.get('participant_name')}"
            )

    @given(
        chunks=st.lists(
            search_result_with_participant_generator(),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=100)
    def test_passages_grouped_under_correct_participant_header(
        self, chunks: list[SearchResult]
    ):
        """Each passage SHALL appear under the header for its participant name.

        **Validates: Requirements 1.1, 1.6**
        """
        # Ensure distinct texts so we can unambiguously locate each passage
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        output = orchestrator._format_context(chunks)

        # Split output into participant sections by the separator
        sections = output.split("\n\n---\n\n")

        # For each section, extract the participant name from its header
        # and verify that only passages belonging to that participant appear in it
        for section in sections:
            # Extract participant name from the section header
            header_match = re.match(
                r"## Conversation with (.+?) \(Your private conversation\)", section
            )
            assert header_match, (
                f"Section does not start with expected header pattern. "
                f"Section start: '{section[:100]}...'"
            )
            section_participant = header_match.group(1)

            # Check each passage: if it belongs to this participant, it must be here
            for chunk in chunks:
                participant = chunk.metadata.get("participant_name", "Unknown")
                if participant == section_participant:
                    assert chunk.text in section, (
                        f"Passage '{chunk.text[:50]}...' belongs to participant "
                        f"'{participant}' but was not found in their section."
                    )

    @given(
        chunks=st.lists(
            search_result_with_participant_generator(),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=100)
    def test_no_passages_discarded(self, chunks: list[SearchResult]):
        """The formatter SHALL NOT discard any input passages.

        **Validates: Requirements 1.6**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        output = orchestrator._format_context(chunks)

        # Every passage text must be present in the output
        for chunk in chunks:
            assert chunk.text in output, (
                f"Passage '{chunk.text[:50]}...' was discarded from output. "
                f"Participant: {chunk.metadata.get('participant_name')}"
            )

    @given(
        chunks=st.lists(
            search_result_with_participant_generator(
                participant_names=["Alice", "Bob", "Charlie"]
            ),
            min_size=2,
            max_size=15,
        )
    )
    @settings(max_examples=100)
    def test_all_participants_have_sections(self, chunks: list[SearchResult]):
        """Every distinct participant in the input SHALL have a section in output.

        **Validates: Requirements 1.1**
        """
        # Ensure we have at least 2 distinct participants
        participants = {c.metadata.get("participant_name", "Unknown") for c in chunks}
        assume(len(participants) >= 2)

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        output = orchestrator._format_context(chunks)

        # Each participant should have their header in the output
        for participant in participants:
            expected_header = (
                f"## Conversation with {participant} (Your private conversation)"
            )
            assert expected_header in output, (
                f"Participant '{participant}' does not have a section header "
                f"in the formatted output."
            )


# --- Property 6: Date Range Annotation Format ---


import re


@st.composite
def date_string_generator(draw):
    """Generate a date string in YYYY-MM-DD format."""
    year = draw(st.integers(min_value=2020, max_value=2025))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # 28 to avoid invalid dates
    return f"{year:04d}-{month:02d}-{day:02d}"


@st.composite
def search_result_with_dates_generator(draw):
    """Generate a SearchResult with both date_range_start and date_range_end metadata."""
    chunk_id = draw(
        st.text(
            min_size=3, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"
        )
    )
    text = draw(
        st.text(
            min_size=5,
            max_size=80,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"), blacklist_characters="\x00"
            ),
        )
    )
    score = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    participant = draw(st.sampled_from(["Alice", "Bob", "Charlie"]))
    start_date = draw(date_string_generator())
    end_date = draw(date_string_generator())
    # Ensure start <= end
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    return SearchResult(
        chunk_id=f"chunk_{chunk_id}",
        text=text,
        metadata={
            "participant_name": participant,
            "source_filename": "file.pdf",
            "date_range_start": start_date,
            "date_range_end": end_date,
        },
        score=score,
    )


@st.composite
def search_result_without_dates_generator(draw):
    """Generate a SearchResult without date_range_start metadata."""
    chunk_id = draw(
        st.text(
            min_size=3, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"
        )
    )
    text = draw(
        st.text(
            min_size=5,
            max_size=80,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"), blacklist_characters="\x00"
            ),
        )
    )
    score = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    participant = draw(st.sampled_from(["Alice", "Bob", "Charlie"]))

    return SearchResult(
        chunk_id=f"chunk_{chunk_id}",
        text=text,
        metadata={
            "participant_name": participant,
            "source_filename": "file.pdf",
        },
        score=score,
    )


# Regex pattern matching [YYYY-MM-DD to YYYY-MM-DD]
_DATE_ANNOTATION_PATTERN = re.compile(r"\[\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\]")


class TestDateRangeAnnotationFormat:
    """Property 6: Date Range Annotation Format.

    # Feature: participant-context-clarity, Property 6: Date Range Annotation Format

    For any SearchResult with both date_range_start and date_range_end metadata
    present, the formatted passage SHALL include an annotation in the format
    [YYYY-MM-DD to YYYY-MM-DD], and passages without date_range_start SHALL have
    no date annotation.
    """

    @given(
        dated_results=st.lists(
            search_result_with_dates_generator(), min_size=1, max_size=5
        ),
    )
    @settings(max_examples=100)
    def test_dated_passages_have_date_annotation(self, dated_results):
        """SearchResults with both date_range_start and date_range_end SHALL
        include an annotation in the format [YYYY-MM-DD to YYYY-MM-DD].

        **Validates: Requirements 1.5**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(dated_results)

        # Each dated result should have its date annotation present in the output
        for result in dated_results:
            start = result.metadata["date_range_start"]
            end = result.metadata["date_range_end"]
            expected_annotation = f"[{start} to {end}]"
            assert expected_annotation in formatted, (
                f"Expected date annotation '{expected_annotation}' not found in "
                f"formatted output. Output:\n{formatted}"
            )

    @given(
        undated_results=st.lists(
            search_result_without_dates_generator(), min_size=1, max_size=5
        ),
    )
    @settings(max_examples=100)
    def test_undated_passages_have_no_date_annotation(self, undated_results):
        """SearchResults without date_range_start SHALL have no date annotation.

        **Validates: Requirements 1.5**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(undated_results)

        # Remove the section headers from consideration (they don't have dates)
        # Check that there are no date annotations in the output
        # Since none of the results have dates, there should be no [YYYY-MM-DD to YYYY-MM-DD] pattern
        assert not _DATE_ANNOTATION_PATTERN.search(formatted), (
            f"Found unexpected date annotation in output for undated passages. "
            f"Output:\n{formatted}"
        )

    @given(
        dated_results=st.lists(
            search_result_with_dates_generator(), min_size=1, max_size=3
        ),
        undated_results=st.lists(
            search_result_without_dates_generator(), min_size=1, max_size=3
        ),
    )
    @settings(max_examples=100)
    def test_mixed_passages_only_dated_have_annotations(
        self, dated_results, undated_results
    ):
        """When mixing dated and undated passages, only dated passages SHALL
        have date annotations and undated passages SHALL have none.

        **Validates: Requirements 1.5**
        """
        all_results = dated_results + undated_results
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(all_results)

        # All dated results should have their annotations
        for result in dated_results:
            start = result.metadata["date_range_start"]
            end = result.metadata["date_range_end"]
            expected_annotation = f"[{start} to {end}]"
            assert expected_annotation in formatted, (
                f"Expected date annotation '{expected_annotation}' not found in "
                f"formatted output with mixed passages."
            )

        # Count annotations: should match number of dated results (accounting for
        # possible duplicate date ranges)
        all_annotations = _DATE_ANNOTATION_PATTERN.findall(formatted)
        # Each annotation should correspond to a dated result
        expected_annotations = set()
        for result in dated_results:
            start = result.metadata["date_range_start"]
            end = result.metadata["date_range_end"]
            expected_annotations.add(f"[{start} to {end}]")

        # Every annotation found should be from a dated result
        for annotation in all_annotations:
            assert annotation in expected_annotations, (
                f"Found annotation '{annotation}' which doesn't correspond to any "
                f"dated result. Expected annotations: {expected_annotations}"
            )

    @given(
        result=search_result_with_dates_generator(),
    )
    @settings(max_examples=100)
    def test_date_annotation_format_matches_pattern(self, result):
        """Date annotations SHALL follow the exact format [YYYY-MM-DD to YYYY-MM-DD].

        **Validates: Requirements 1.5**
        """
        # Test _format_date_label directly
        label = RAGOrchestrator._format_date_label(result)

        assert (
            label is not None
        ), "Expected a date label for a result with both date_range_start and date_range_end"
        assert _DATE_ANNOTATION_PATTERN.fullmatch(
            label
        ), f"Date label '{label}' does not match expected format [YYYY-MM-DD to YYYY-MM-DD]"

    @given(
        result=search_result_without_dates_generator(),
    )
    @settings(max_examples=100)
    def test_format_date_label_returns_none_for_undated(self, result):
        """_format_date_label SHALL return None for passages without date metadata.

        **Validates: Requirements 1.5**
        """
        label = RAGOrchestrator._format_date_label(result)

        assert label is None, f"Expected None for undated result, got '{label}'"


# --- Property 3: Section Separation ---


@st.composite
def multi_participant_chunks_generator(draw, min_participants=2, max_participants=5):
    """Generate SearchResult lists guaranteed to have multiple distinct participants.

    Each participant gets 1-3 passages with unique texts. Participants have
    distinct scores to ensure deterministic ordering.
    """
    all_participants = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    num_participants = draw(
        st.integers(min_value=min_participants, max_value=max_participants)
    )
    participants = draw(
        st.lists(
            st.sampled_from(all_participants),
            min_size=num_participants,
            max_size=num_participants,
            unique=True,
        )
    )

    results = []
    for idx, participant in enumerate(participants):
        num_passages = draw(st.integers(min_value=1, max_value=3))
        for i in range(num_passages):
            score = draw(
                st.floats(
                    min_value=0.01,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            text = draw(
                st.text(
                    min_size=5,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("L", "N", "Z"),
                        blacklist_characters="\x00\n",
                    ),
                ).filter(lambda s: s.strip())
            )
            has_dates = draw(st.booleans())
            metadata: dict = {
                "participant_name": participant,
                "source_filename": f"{participant.lower()}.pdf",
            }
            if has_dates:
                metadata["date_range_start"] = f"2024-{(idx + 1):02d}-{(i + 1):02d}"
                metadata["date_range_end"] = f"2024-{(idx + 1):02d}-{(i + 2):02d}"

            results.append(
                SearchResult(
                    chunk_id=f"chunk_{participant}_{i}",
                    text=text,
                    metadata=metadata,
                    score=score,
                )
            )

    return results


class TestSectionSeparation:
    """Property 3: Section Separation.

    # Feature: participant-context-clarity, Property 3: Section Separation

    For any set of SearchResults containing passages from two or more distinct
    participants, the formatted output SHALL contain a `---` separator between
    each pair of adjacent participant sections, and SHALL NOT contain a separator
    before the first section or after the last section.

    **Validates: Requirements 1.3**
    """

    @given(chunks=multi_participant_chunks_generator(min_participants=2))
    @settings(max_examples=100)
    def test_separator_between_adjacent_sections(self, chunks):
        """A `---` separator SHALL exist between each pair of adjacent
        participant sections.

        **Validates: Requirements 1.3**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Count distinct participants in input
        participants = {c.metadata.get("participant_name", "Unknown") for c in chunks}
        num_participants = len(participants)

        # Split by the separator pattern to get sections
        sections = formatted.split("\n\n---\n\n")

        # Number of sections should equal number of distinct participants
        assert len(sections) == num_participants, (
            f"Expected {num_participants} sections (one per participant) "
            f"but got {len(sections)} when splitting by '---' separator.\n"
            f"Participants: {participants}\n"
            f"Output:\n{formatted}"
        )

    @given(chunks=multi_participant_chunks_generator(min_participants=2))
    @settings(max_examples=100)
    def test_no_separator_before_first_section(self, chunks):
        """The formatted output SHALL NOT start with a `---` separator.

        **Validates: Requirements 1.3**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Output must not start with the separator or whitespace+separator
        assert not formatted.lstrip().startswith("---"), (
            f"Formatted output starts with a separator, which should not happen.\n"
            f"Output starts with: '{formatted[:50]}...'"
        )

    @given(chunks=multi_participant_chunks_generator(min_participants=2))
    @settings(max_examples=100)
    def test_no_separator_after_last_section(self, chunks):
        """The formatted output SHALL NOT end with a `---` separator.

        **Validates: Requirements 1.3**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Output must not end with the separator or separator+whitespace
        assert not formatted.rstrip().endswith("---"), (
            f"Formatted output ends with a separator, which should not happen.\n"
            f"Output ends with: '...{formatted[-50:]}'"
        )

    @given(chunks=multi_participant_chunks_generator(min_participants=2))
    @settings(max_examples=100)
    def test_separator_count_matches_section_boundaries(self, chunks):
        """The number of `---` separators SHALL equal the number of distinct
        participants minus one (exactly one separator between each pair of
        adjacent sections).

        **Validates: Requirements 1.3**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Count distinct participants
        participants = {c.metadata.get("participant_name", "Unknown") for c in chunks}
        num_participants = len(participants)
        expected_separators = num_participants - 1

        # Count occurrences of the separator pattern
        separator_count = formatted.count("\n\n---\n\n")

        assert separator_count == expected_separators, (
            f"Expected {expected_separators} separators (for {num_participants} "
            f"participant sections) but found {separator_count}.\n"
            f"Participants: {participants}\n"
            f"Output:\n{formatted}"
        )

    @given(
        chunks=st.lists(
            search_result_with_participant_generator(participant_names=["Alice"]),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_single_participant_has_no_separator(self, chunks):
        """When only one participant exists, the formatted output SHALL NOT
        contain any `---` separator.

        **Validates: Requirements 1.3**
        """
        # Ensure all chunks belong to same participant
        assume(len({c.metadata.get("participant_name") for c in chunks}) == 1)

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        assert "---" not in formatted, (
            f"Single-participant output should not contain any '---' separator.\n"
            f"Output:\n{formatted}"
        )


# --- Property 4: Chronological Ordering Within Participant Sections ---


@st.composite
def same_participant_dated_results_generator(draw, participant="Alice"):
    """Generate SearchResults for a single participant with varying dates.

    Produces a mix of dated and undated results for the same participant,
    ensuring at least 2 dated results for meaningful ordering verification.
    """
    # Generate 2-6 dated results with distinct start dates
    num_dated = draw(st.integers(min_value=2, max_value=6))
    dated_results = []
    for i in range(num_dated):
        year = 2024
        month = draw(st.integers(min_value=1, max_value=12))
        day = draw(st.integers(min_value=1, max_value=28))
        start_date = f"{year:04d}-{month:02d}-{day:02d}"
        # End date is same or later
        end_day = draw(st.integers(min_value=day, max_value=28))
        end_date = f"{year:04d}-{month:02d}-{end_day:02d}"

        text = draw(
            st.text(
                min_size=5,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "Z"),
                    blacklist_characters="\x00\n",
                ),
            ).filter(lambda s: s.strip())
        )
        score = draw(
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        dated_results.append(
            SearchResult(
                chunk_id=f"chunk_dated_{i}",
                text=text,
                metadata={
                    "participant_name": participant,
                    "source_filename": "test.pdf",
                    "date_range_start": start_date,
                    "date_range_end": end_date,
                },
                score=score,
            )
        )

    # Optionally add 0-3 undated results
    num_undated = draw(st.integers(min_value=0, max_value=3))
    undated_results = []
    for i in range(num_undated):
        text = draw(
            st.text(
                min_size=5,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "Z"),
                    blacklist_characters="\x00\n",
                ),
            ).filter(lambda s: s.strip())
        )
        score = draw(
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        undated_results.append(
            SearchResult(
                chunk_id=f"chunk_undated_{i}",
                text=text,
                metadata={
                    "participant_name": participant,
                    "source_filename": "test.pdf",
                },
                score=score,
            )
        )

    # Shuffle to ensure the formatter must sort them
    all_results = dated_results + undated_results
    shuffled = draw(st.permutations(all_results))
    return list(shuffled)


class TestChronologicalOrderingWithinParticipantSections:
    """Property 4: Chronological Ordering Within Participant Sections.

    # Feature: participant-context-clarity, Property 4: Chronological Ordering Within Participant Sections

    For any set of SearchResults belonging to the same participant with varying
    date_range_start metadata values, the formatted output SHALL present dated
    passages in ascending chronological order, with undated passages placed after
    all dated passages.

    **Validates: Requirements 1.4, 1.7**
    """

    @given(chunks=same_participant_dated_results_generator())
    @settings(max_examples=100)
    def test_dated_passages_in_ascending_order(self, chunks):
        """Dated passages within a participant section SHALL be ordered in
        ascending chronological order by date_range_start.

        **Validates: Requirements 1.4**
        """
        # Ensure texts are distinct so we can find them in output
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Extract dated chunks and their expected order
        dated_chunks = [c for c in chunks if c.metadata.get("date_range_start")]
        expected_order = sorted(
            dated_chunks, key=lambda c: c.metadata["date_range_start"]
        )

        # Find positions of each dated passage text in the formatted output
        positions = []
        for chunk in expected_order:
            pos = formatted.find(chunk.text)
            assert (
                pos != -1
            ), f"Dated passage '{chunk.text[:40]}...' not found in formatted output."
            positions.append((pos, chunk.metadata["date_range_start"], chunk.text[:40]))

        # Verify positions are monotonically increasing (ascending order)
        for i in range(len(positions) - 1):
            assert positions[i][0] < positions[i + 1][0], (
                f"Dated passages not in ascending chronological order.\n"
                f"Passage at date '{positions[i][1]}' (pos {positions[i][0]}) "
                f"appears after passage at date '{positions[i + 1][1]}' "
                f"(pos {positions[i + 1][0]}).\n"
                f"Text at earlier date: '{positions[i][2]}...'\n"
                f"Text at later date: '{positions[i + 1][2]}...'"
            )

    @given(chunks=same_participant_dated_results_generator())
    @settings(max_examples=100)
    def test_undated_passages_placed_after_dated(self, chunks):
        """Undated passages SHALL be placed after all dated passages within
        the same participant section.

        **Validates: Requirements 1.7**
        """
        # Ensure texts are distinct
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        # Need at least one undated passage for this test to be meaningful
        undated = [c for c in chunks if not c.metadata.get("date_range_start")]
        dated = [c for c in chunks if c.metadata.get("date_range_start")]
        assume(len(undated) >= 1 and len(dated) >= 1)

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Find the last position of any dated passage
        last_dated_pos = -1
        for chunk in dated:
            pos = formatted.find(chunk.text)
            assert (
                pos != -1
            ), f"Dated passage '{chunk.text[:40]}...' not found in output."
            last_dated_pos = max(last_dated_pos, pos)

        # Find the first position of any undated passage
        first_undated_pos = len(formatted)
        for chunk in undated:
            pos = formatted.find(chunk.text)
            assert (
                pos != -1
            ), f"Undated passage '{chunk.text[:40]}...' not found in output."
            first_undated_pos = min(first_undated_pos, pos)

        # All undated passages must appear after all dated passages
        assert first_undated_pos > last_dated_pos, (
            f"Undated passage appears before a dated passage.\n"
            f"Last dated passage ends at position {last_dated_pos}, "
            f"but first undated passage starts at position {first_undated_pos}."
        )

    @given(
        chunks=st.lists(
            search_result_with_dates_generator(),
            min_size=2,
            max_size=8,
        )
    )
    @settings(max_examples=100)
    def test_chronological_order_with_same_participant_dated_only(self, chunks):
        """When all passages for a participant have dates, they SHALL appear
        in ascending date_range_start order.

        **Validates: Requirements 1.4**
        """
        # Force all chunks to same participant so they end up in one section
        forced_chunks = [
            SearchResult(
                chunk_id=c.chunk_id,
                text=c.text,
                metadata={
                    **c.metadata,
                    "participant_name": "Alice",
                },
                score=c.score,
            )
            for c in chunks
        ]
        # Ensure distinct texts
        texts = [c.text for c in forced_chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(forced_chunks)

        # Expected order: ascending by date_range_start
        expected_order = sorted(
            forced_chunks, key=lambda c: c.metadata["date_range_start"]
        )

        # Verify positions in output match expected order
        prev_pos = -1
        for chunk in expected_order:
            pos = formatted.find(chunk.text)
            assert pos != -1, f"Passage '{chunk.text[:40]}...' not found in output."
            assert pos > prev_pos, (
                f"Passage with date '{chunk.metadata['date_range_start']}' "
                f"appears at position {pos} which is not after previous position "
                f"{prev_pos}. Passages are not in chronological order."
            )
            prev_pos = pos


# --- Property 16: Backward-Compatible Chunk Presentation ---

# Speaker role prefix patterns that would indicate new-format chunks
_SPEAKER_PREFIX_PATTERN = re.compile(r"\[(?:You|Message|[^\]]+)\]: ")


@st.composite
def plain_text_generator(draw):
    """Generate text that does NOT contain any speaker role prefix patterns.

    Produces text without `[You]: `, `[Message]: `, or `[Name]: ` patterns
    to simulate old-format chunks that predate speaker role tagging.
    """
    text = draw(
        st.text(
            min_size=10,
            max_size=120,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z", "P"),
                blacklist_characters="\x00[]",
            ),
        )
    )
    # Ensure no accidental prefix patterns and non-empty
    assume(text.strip())
    assume(not _SPEAKER_PREFIX_PATTERN.search(text))
    return text


@st.composite
def search_result_without_prefix_generator(draw):
    """Generate a SearchResult whose text lacks any speaker prefix patterns.

    Simulates old-format chunks (pre-speaker-role) that should pass through
    the Context_Formatter verbatim.
    """
    chunk_id = draw(
        st.text(
            min_size=3,
            max_size=10,
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        ).map(lambda s: f"chunk_{s}")
    )
    text = draw(plain_text_generator())
    score = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    participant = draw(st.sampled_from(["Alice", "Bob", "Charlie", "Diana"]))

    # Optionally include date metadata
    has_dates = draw(st.booleans())
    metadata: dict = {
        "participant_name": participant,
        "source_filename": f"{participant.lower()}.pdf",
    }
    if has_dates:
        metadata["date_range_start"] = "2024-01-01"
        metadata["date_range_end"] = "2024-01-15"

    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, score=score)


class TestBackwardCompatibleChunkPresentation:
    """Property 16: Backward-Compatible Chunk Presentation.

    # Feature: participant-context-clarity, Property 16: Backward-Compatible Chunk Presentation

    For any SearchResult whose text field does not contain any speaker role
    prefix patterns, the Context_Formatter SHALL include that text verbatim
    in the output without injecting or stripping prefix characters.

    **Validates: Requirements 6.1**
    """

    @given(
        chunks=st.lists(
            search_result_without_prefix_generator(),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_text_passes_through_verbatim(self, chunks):
        """SearchResults without speaker prefix patterns SHALL have their
        text presented verbatim in the formatted output.

        **Validates: Requirements 6.1**
        """
        # Ensure distinct texts for unambiguous lookup
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        for chunk in chunks:
            assert chunk.text in formatted, (
                f"Plain-text passage '{chunk.text[:50]}...' was not found verbatim "
                f"in formatted output. The formatter may have injected or stripped "
                f"characters.\nOutput:\n{formatted}"
            )

    @given(
        chunks=st.lists(
            search_result_without_prefix_generator(),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_no_prefix_injected_into_plain_text(self, chunks):
        """The Context_Formatter SHALL NOT inject speaker role prefixes into
        text that originally lacks them.

        **Validates: Requirements 6.1**
        """
        # Ensure distinct texts
        texts = [c.text for c in chunks]
        assume(len(set(texts)) == len(texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # For each chunk, find its text in the output and verify no prefix
        # was injected immediately before it (other than date labels or headers)
        for chunk in chunks:
            pos = formatted.find(chunk.text)
            assert pos != -1, f"Passage '{chunk.text[:40]}...' not found in output."

            # Check the characters immediately before the text position.
            # The text should NOT be preceded by a speaker prefix pattern
            # (e.g., "[You]: ", "[Message]: ", "[Name]: ") on the same line.
            # Look back up to 50 chars for a prefix that was injected.
            lookback_start = max(0, pos - 50)
            preceding = formatted[lookback_start:pos]
            # Get the last line fragment before the text
            last_line = (
                preceding.rsplit("\n", 1)[-1] if "\n" in preceding else preceding
            )

            # The last line fragment should NOT match a speaker prefix pattern
            assert not _SPEAKER_PREFIX_PATTERN.search(last_line), (
                f"A speaker role prefix was injected before plain-text passage "
                f"'{chunk.text[:40]}...'.\n"
                f"Preceding content on same line: '{last_line}'\n"
                f"The formatter should not inject prefixes into old-format text."
            )

    @given(chunk=search_result_without_prefix_generator())
    @settings(max_examples=100)
    def test_text_not_stripped_or_modified(self, chunk):
        """The Context_Formatter SHALL NOT strip or modify any characters
        from text that lacks speaker prefix patterns.

        **Validates: Requirements 6.1**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context([chunk])

        # The exact text should be present — not a trimmed or modified version
        assert chunk.text in formatted, (
            f"Text was modified or stripped by the formatter.\n"
            f"Original text: '{chunk.text}'\n"
            f"Formatted output:\n{formatted}"
        )

        # Verify the text appears as a contiguous substring (not split across lines
        # that weren't in the original)
        pos = formatted.find(chunk.text)
        extracted = formatted[pos : pos + len(chunk.text)]
        assert extracted == chunk.text, (
            f"Text was modified in the formatted output.\n"
            f"Original: '{chunk.text}'\n"
            f"Extracted: '{extracted}'"
        )


# --- Generators for Property 18 ---


@st.composite
def context_with_speaker_prefix_generator(draw):
    """Generate formatted context strings containing at least one speaker prefix.

    Produces SearchResult lists where at least one chunk has text containing
    a speaker role prefix pattern ([You]: or [{Name}]:).
    """
    # Generate a participant name (non-empty, no brackets or colons)
    participant_name = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Zs"),
                blacklist_characters="[]:|\n",
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda s: s.strip() and not s.isspace())
    )

    # Generate message text
    message_text = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Zs", "P"),
                blacklist_characters="\n",
            ),
            min_size=1,
            max_size=100,
        ).filter(lambda s: s.strip())
    )

    # Choose a prefix pattern type
    prefix_type = draw(st.sampled_from(["you", "name"]))
    if prefix_type == "you":
        prefixed_line = f"[You]: {message_text}"
    else:
        prefixed_line = f"[{participant_name}]: {message_text}"

    # Optionally add more lines (some with prefixes, some without)
    extra_lines = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "Zs", "P"),
                    blacklist_characters="\n",
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda s: s.strip()),
            min_size=0,
            max_size=5,
        )
    )

    # Build the chunk text with at least one prefixed line
    all_lines = [prefixed_line] + [f"[Message]: {line}" for line in extra_lines]
    chunk_text = "\n".join(all_lines)

    # Create SearchResult with prefixed text
    score = draw(st.floats(min_value=0.1, max_value=1.0))
    chunk = SearchResult(
        chunk_id=f"chunk_{draw(st.integers(min_value=1, max_value=9999))}",
        text=chunk_text,
        metadata={"participant_name": participant_name},
        score=score,
    )

    return chunk, participant_name


# --- Property 18: Conditional Speaker Prefix Prompt Instruction ---


class TestConditionalSpeakerPrefixPromptInstruction:
    """Property 18: Conditional Speaker Prefix Prompt Instruction.

    # Feature: participant-context-clarity, Property 18: Conditional Speaker Prefix Prompt Instruction

    *For any* formatted context string that contains at least one line matching
    a speaker role prefix pattern ([You]: or [{Name}]:), the system prompt SHALL
    include an explanation of what the prefixes mean.

    **Validates: Requirements 7.3**
    """

    @given(data=context_with_speaker_prefix_generator())
    @settings(max_examples=100)
    def test_system_prompt_explains_you_prefix(self, data):
        """When context contains speaker prefix patterns, the system prompt
        SHALL explain that [You]: indicates messages sent by the user.

        **Validates: Requirements 7.3**
        """
        chunk, participant_name = data

        # Verify the chunk text actually contains a prefix pattern
        assert re.search(r"\[You\]:", chunk.text) or re.search(
            r"\[[^\]]+\]:", chunk.text
        )

        # The system prompt must explain [You]: prefix meaning
        assert "[You]:" in _SYSTEM_TEMPLATE, (
            "System prompt must explain the [You]: prefix when context "
            "contains speaker role prefix patterns."
        )
        assert (
            "sent by the user" in _SYSTEM_TEMPLATE.lower()
            or "sent by the user" in _SYSTEM_TEMPLATE
        ), (
            "System prompt must explain that [You]: indicates messages "
            "sent by the user."
        )

    @given(data=context_with_speaker_prefix_generator())
    @settings(max_examples=100)
    def test_system_prompt_explains_participant_prefix(self, data):
        """When context contains speaker prefix patterns, the system prompt
        SHALL explain that [{Name}]: indicates messages received from
        the participant.

        **Validates: Requirements 7.3**
        """
        chunk, participant_name = data

        # Verify the chunk text actually contains a prefix pattern
        assert re.search(r"\[You\]:", chunk.text) or re.search(
            r"\[[^\]]+\]:", chunk.text
        )

        # The system prompt must explain participant name prefix meaning
        assert (
            "received from" in _SYSTEM_TEMPLATE.lower()
            or "received from" in _SYSTEM_TEMPLATE
        ), (
            "System prompt must explain that [{Name}]: indicates messages "
            "received from that participant."
        )

    @given(data=context_with_speaker_prefix_generator())
    @settings(max_examples=100)
    def test_system_prompt_explains_message_prefix(self, data):
        """When context contains speaker prefix patterns, the system prompt
        SHALL explain that [Message]: indicates unknown direction.

        **Validates: Requirements 7.3**
        """
        chunk, participant_name = data

        # The system prompt must explain [Message]: prefix meaning
        assert "[Message]:" in _SYSTEM_TEMPLATE, (
            "System prompt must explain the [Message]: prefix for unknown "
            "direction messages."
        )
        assert (
            "unknown direction" in _SYSTEM_TEMPLATE.lower()
            or "unknown direction" in _SYSTEM_TEMPLATE
        ), (
            "System prompt must explain that [Message]: indicates " "unknown direction."
        )

    @given(data=context_with_speaker_prefix_generator())
    @settings(max_examples=100)
    def test_system_prompt_included_when_context_has_prefixes(self, data):
        """When formatted context contains speaker prefixes, the system
        prompt with prefix explanations SHALL be part of the prompt template
        used by the orchestrator.

        **Validates: Requirements 7.3**
        """
        chunk, participant_name = data

        # Create an orchestrator and verify it uses the system template
        # that includes prefix explanations
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
        )

        # The orchestrator's prompt template should include the system message
        # that explains prefixes
        prompt_messages = orchestrator._prompt_template.format_messages(
            context=chunk.text,
            query="test query",
        )

        # Extract the system message
        system_messages = [msg for msg in prompt_messages if msg.type == "system"]
        assert (
            len(system_messages) == 1
        ), "Orchestrator prompt template must include exactly one system message."

        system_content = system_messages[0].content

        # Verify the system message includes prefix explanations
        assert (
            "[You]:" in system_content
        ), "System message in prompt must mention [You]: prefix."
        assert (
            "received from" in system_content.lower()
            or "received from" in system_content
        ), "System message in prompt must explain name prefix meaning."

    @given(data=context_with_speaker_prefix_generator())
    @settings(max_examples=100)
    def test_formatted_context_preserves_prefix_patterns(self, data):
        """The Context_Formatter SHALL preserve speaker prefix patterns
        in the formatted output when they exist in the source text.

        **Validates: Requirements 7.3**
        """
        chunk, participant_name = data

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context([chunk])

        # The formatted context should still contain the prefix patterns
        # from the original chunk text
        has_you_prefix = "[You]:" in chunk.text
        has_name_prefix = bool(
            re.search(r"\[[^\]]+\]:", chunk.text)
            and "[You]:" not in chunk.text
            and "[Message]:" not in chunk.text
        ) or (re.search(r"\[[^\]]+\]:", chunk.text) and "[You]:" in chunk.text)

        if has_you_prefix:
            assert (
                "[You]:" in formatted
            ), "Formatted context must preserve [You]: prefix patterns."

        # The chunk text itself should appear in the formatted output
        assert chunk.text in formatted, (
            "Formatted context must include the original chunk text "
            "containing speaker prefixes."
        )


# --- Task 9.1: Mixed-Format Context Backward Compatibility ---
# Validates: Requirements 6.1, 6.2, 6.5


@st.composite
def prefixed_text_generator(draw):
    """Generate text that contains speaker role prefix patterns (new-format chunks).

    Simulates chunks produced after speaker role tagging was implemented.
    """
    participant_name = draw(
        st.sampled_from(["Alice", "Bob", "Charlie", "Diana", "Kyriaki Salavanitou"])
    )
    # Generate 1-5 prefixed message lines
    num_lines = draw(st.integers(min_value=1, max_value=5))
    lines = []
    for _ in range(num_lines):
        msg_text = draw(
            st.text(
                min_size=3,
                max_size=60,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "Z"),
                    blacklist_characters="\x00\n[]:",
                ),
            ).filter(lambda s: s.strip())
        )
        prefix_type = draw(st.sampled_from(["you", "name", "message"]))
        if prefix_type == "you":
            lines.append(f"[You]: {msg_text}")
        elif prefix_type == "name":
            lines.append(f"[{participant_name}]: {msg_text}")
        else:
            lines.append(f"[Message]: {msg_text}")

    return "\n".join(lines), participant_name


@st.composite
def mixed_format_search_results_generator(draw):
    """Generate a list of SearchResults with a mix of old-format (plain text)
    and new-format (speaker-prefixed) chunks.

    Ensures at least one of each format is present.
    """
    # Generate 1-4 old-format (plain text) chunks
    num_old = draw(st.integers(min_value=1, max_value=4))
    old_chunks = []
    for i in range(num_old):
        text = draw(plain_text_generator())
        participant = draw(st.sampled_from(["Alice", "Bob", "Charlie"]))
        score = draw(
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        has_dates = draw(st.booleans())
        metadata: dict = {
            "participant_name": participant,
            "source_filename": f"{participant.lower()}.pdf",
        }
        if has_dates:
            metadata["date_range_start"] = f"2024-01-{(i + 1):02d}"
            metadata["date_range_end"] = f"2024-01-{(i + 2):02d}"
        old_chunks.append(
            SearchResult(
                chunk_id=f"old_chunk_{i}",
                text=text,
                metadata=metadata,
                score=score,
            )
        )

    # Generate 1-4 new-format (prefixed) chunks
    num_new = draw(st.integers(min_value=1, max_value=4))
    new_chunks = []
    for i in range(num_new):
        prefixed_text, participant_name = draw(prefixed_text_generator())
        score = draw(
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        has_dates = draw(st.booleans())
        metadata: dict = {
            "participant_name": participant_name,
            "source_filename": f"{participant_name.lower().replace(' ', '_')}.pdf",
        }
        if has_dates:
            metadata["date_range_start"] = f"2024-02-{(i + 1):02d}"
            metadata["date_range_end"] = f"2024-02-{(i + 2):02d}"
        new_chunks.append(
            SearchResult(
                chunk_id=f"new_chunk_{i}",
                text=prefixed_text,
                metadata=metadata,
                score=score,
            )
        )

    # Shuffle both together
    all_chunks = old_chunks + new_chunks
    shuffled = draw(st.permutations(all_chunks))
    return list(shuffled)


class TestMixedFormatContextBackwardCompatibility:
    """Task 9.1: Verify orchestrator handles mixed-format context without errors.

    Tests that `_format_context` works when some SearchResults have prefixed text
    (speaker role prefixes like [You]:, [Name]:, [Message]:) and others have
    plain text without any prefixes.

    Tests that `query()` returns valid GenerationResult with non-empty text
    and non-empty source_chunks regardless of prefix presence.

    **Validates: Requirements 6.1, 6.2, 6.5**
    """

    @given(chunks=mixed_format_search_results_generator())
    @settings(max_examples=100)
    def test_format_context_handles_mixed_formats_without_error(self, chunks):
        """_format_context SHALL process a mix of old-format and new-format
        chunks without raising any errors.

        **Validates: Requirements 6.5**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        # Should not raise any exception
        formatted = orchestrator._format_context(chunks)

        # Output should be a non-empty string
        assert isinstance(formatted, str)
        assert (
            len(formatted) > 0
        ), "Formatted context should be non-empty when given mixed-format chunks."

    @given(chunks=mixed_format_search_results_generator())
    @settings(max_examples=100)
    def test_old_chunks_pass_through_verbatim_in_mixed_context(self, chunks):
        """Old chunks (without prefixes) SHALL pass through verbatim in their
        participant section even when mixed with new-format chunks.

        **Validates: Requirements 6.1**
        """
        # Identify old-format chunks (those without speaker prefix patterns)
        old_chunks = [c for c in chunks if not _SPEAKER_PREFIX_PATTERN.search(c.text)]
        assume(len(old_chunks) >= 1)

        # Ensure distinct texts for unambiguous lookup
        all_texts = [c.text for c in chunks]
        assume(len(set(all_texts)) == len(all_texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Each old-format chunk text should appear verbatim
        for chunk in old_chunks:
            assert chunk.text in formatted, (
                f"Old-format chunk text '{chunk.text[:50]}...' was not found "
                f"verbatim in mixed-format output. The formatter may have "
                f"modified it.\nOutput:\n{formatted[:500]}"
            )

    @given(chunks=mixed_format_search_results_generator())
    @settings(max_examples=100)
    def test_new_chunks_preserved_in_mixed_context(self, chunks):
        """New chunks (with speaker prefixes) SHALL also be preserved in
        the formatted output alongside old chunks.

        **Validates: Requirements 6.5**
        """
        # Identify new-format chunks (those with speaker prefix patterns)
        new_chunks = [c for c in chunks if _SPEAKER_PREFIX_PATTERN.search(c.text)]
        assume(len(new_chunks) >= 1)

        # Ensure distinct texts
        all_texts = [c.text for c in chunks]
        assume(len(set(all_texts)) == len(all_texts))

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Each new-format chunk text should appear in the output
        for chunk in new_chunks:
            assert chunk.text in formatted, (
                f"New-format chunk text '{chunk.text[:50]}...' was not found "
                f"in mixed-format output.\nOutput:\n{formatted[:500]}"
            )

    @given(chunks=mixed_format_search_results_generator())
    @settings(max_examples=100)
    def test_no_prefix_injected_into_old_chunks_in_mixed_context(self, chunks):
        """The Context_Formatter SHALL NOT inject speaker role prefixes into
        old-format text that originally lacks them, even when new-format
        chunks are present in the same context.

        **Validates: Requirements 6.1**
        """
        old_chunks = [c for c in chunks if not _SPEAKER_PREFIX_PATTERN.search(c.text)]
        assume(len(old_chunks) >= 1)

        # Ensure distinct texts and that no old chunk text is a substring
        # of any other chunk text (avoids false positives from find())
        all_texts = [c.text for c in chunks]
        assume(len(set(all_texts)) == len(all_texts))
        for old_c in old_chunks:
            for other_c in chunks:
                if other_c.chunk_id != old_c.chunk_id:
                    assume(old_c.text not in other_c.text)

        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        for chunk in old_chunks:
            pos = formatted.find(chunk.text)
            assert (
                pos != -1
            ), f"Old-format chunk '{chunk.text[:40]}...' not found in output."

            # Check that no speaker prefix was injected on the line before this text
            lookback_start = max(0, pos - 50)
            preceding = formatted[lookback_start:pos]
            last_line = (
                preceding.rsplit("\n", 1)[-1] if "\n" in preceding else preceding
            )

            assert not _SPEAKER_PREFIX_PATTERN.search(last_line), (
                f"A speaker role prefix was injected before old-format chunk "
                f"'{chunk.text[:40]}...' in mixed context.\n"
                f"Preceding content on same line: '{last_line}'"
            )

    @given(
        chunks=mixed_format_search_results_generator(),
        llm_response=llm_response_generator(),
    )
    @settings(max_examples=100)
    def test_query_returns_valid_generation_result_with_mixed_chunks(
        self, chunks, llm_response
    ):
        """query() SHALL return a valid GenerationResult with non-empty text
        and non-empty source_chunks regardless of whether retrieved context
        contains speaker role prefixes, lacks them, or has a mix.

        **Validates: Requirements 6.2**
        """
        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider(llm_response)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=20,
        )

        result = orchestrator.query("What did Alice say?")

        # Result must be a valid GenerationResult
        assert isinstance(result, GenerationResult)
        assert (
            result.text
        ), "GenerationResult.text should be non-empty with mixed-format chunks."
        assert len(result.text) > 0
        assert result.source_chunks, (
            "GenerationResult.source_chunks should be non-empty "
            "with mixed-format chunks."
        )
        assert len(result.source_chunks) > 0

    @given(
        chunks=st.lists(
            search_result_without_prefix_generator(),
            min_size=1,
            max_size=8,
        ),
        llm_response=llm_response_generator(),
    )
    @settings(max_examples=100)
    def test_query_returns_valid_result_with_only_old_chunks(
        self, chunks, llm_response
    ):
        """query() SHALL return valid GenerationResult even when ALL chunks
        are old-format (no speaker prefixes).

        **Validates: Requirements 6.2**
        """
        mock_retriever = _create_mock_retriever(chunks)
        mock_llm = _create_mock_llm_provider(llm_response)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_chunks=20,
        )

        result = orchestrator.query("Tell me about the conversation")

        assert isinstance(result, GenerationResult)
        assert (
            result.text
        ), "GenerationResult.text should be non-empty with old-format chunks."
        assert result.source_chunks, (
            "GenerationResult.source_chunks should be non-empty "
            "with old-format chunks."
        )

    @given(chunks=mixed_format_search_results_generator())
    @settings(max_examples=100)
    def test_all_chunks_included_in_mixed_format_output(self, chunks):
        """_format_context SHALL include all input passages without discarding
        any, regardless of whether they have prefixes or not.

        **Validates: Requirements 6.5**
        """
        orchestrator = RAGOrchestrator(
            retriever=MagicMock(),
            llm_provider=MagicMock(),
        )

        formatted = orchestrator._format_context(chunks)

        # Every chunk text must appear in the formatted output
        for chunk in chunks:
            assert chunk.text in formatted, (
                f"Chunk '{chunk.text[:50]}...' (format: "
                f"{'new' if _SPEAKER_PREFIX_PATTERN.search(chunk.text) else 'old'}) "
                f"was discarded from mixed-format output."
            )
