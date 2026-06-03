"""Property-based tests for the ConversationChunker.

# Feature: sms-rag, Property 5: Chunking Structural Correctness
# Feature: sms-rag, Property 6: Chunk Metadata Completeness

Validates: Requirements 3.1, 3.2, 3.3

Property 5: For any list of N messages and configurable chunk_size C and
overlap O (where O < C), the chunker SHALL produce chunks such that:
(a) each chunk contains at most C messages,
(b) no single message is split across chunks,
(c) for any two adjacent chunks i and i+1, the last O messages of chunk i
    are identical to the first O messages of chunk i+1, and
(d) the union of all chunks covers all N messages.

Property 6: For any ConversationChunk produced by the chunker, the chunk SHALL
have non-empty participant_name and source_filename fields, and its
date_range_start/date_range_end SHALL span the timestamps of the constituent
messages (when timestamps are present).
"""

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.shared.models import Message, ParsedConversation


# --- Generators ---


@st.composite
def message_generator(draw):
    """Generate a random Message object with optional timestamps, types, and phone numbers."""
    text = draw(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_characters="\x00",
            ),
        )
    )

    # Optional timestamp
    has_timestamp = draw(st.booleans())
    timestamp = None
    if has_timestamp:
        timestamp = draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2025, 12, 31),
            )
        )

    # Optional message type
    message_type = draw(
        st.one_of(
            st.none(),
            st.sampled_from(["SMS", "iMessage", "RCS"]),
        )
    )

    # Optional phone number
    phone_number = draw(
        st.one_of(
            st.none(),
            st.from_regex(r"\+\d{10,12}", fullmatch=True),
        )
    )

    return Message(
        text=text,
        timestamp=timestamp,
        message_type=message_type,
        phone_number=phone_number,
    )


@st.composite
def conversation_generator(draw, min_messages=1, max_messages=100):
    """Generate a random ParsedConversation with configurable message counts."""
    num_messages = draw(st.integers(min_value=min_messages, max_value=max_messages))
    messages = draw(
        st.lists(message_generator(), min_size=num_messages, max_size=num_messages)
    )

    participant_name = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00",
            ),
        )
    )

    source_filename = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                blacklist_characters="\x00",
            ),
        ).map(lambda s: s + ".pdf")
    )

    return ParsedConversation(
        participant_name=participant_name,
        source_filename=source_filename,
        messages=messages,
    )


@st.composite
def chunk_config_generator(draw):
    """Generate random valid (chunk_size, overlap) pairs where overlap < chunk_size."""
    chunk_size = draw(st.integers(min_value=1, max_value=50))
    overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return chunk_size, overlap


# --- Property 5: Chunking Structural Correctness ---


class TestChunkingStructuralCorrectness:
    """Property 5: Chunking Structural Correctness.

    # Feature: sms-rag, Property 5: Chunking Structural Correctness
    """

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_each_chunk_has_at_most_chunk_size_messages(self, conversation, config):
        """(a) Each chunk contains at most C messages.

        **Validates: Requirements 3.1**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            assert chunk.message_count <= chunk_size, (
                f"Chunk has {chunk.message_count} messages, "
                f"exceeds chunk_size={chunk_size}"
            )

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_no_message_split_across_chunks(self, conversation, config):
        """(b) No single message is split across chunks.

        Each message text appears as a complete line in the chunk text,
        never partially split.

        **Validates: Requirements 3.1**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            # Each line in the chunk text should correspond to a complete message
            lines = chunk.text.split("\n")
            # The number of lines should equal the message_count
            assert (
                len(lines) == chunk.message_count
            ), f"Chunk has {len(lines)} lines but message_count={chunk.message_count}"
            # Each line should be a complete message text from the conversation
            for line in lines:
                assert any(
                    msg.text == line for msg in conversation.messages
                ), f"Line '{line[:50]}...' not found as a complete message"

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_adjacent_chunks_overlap_correctly(self, conversation, config):
        """(c) For any two adjacent chunks i and i+1, the last O messages of
        chunk i are identical to the first O messages of chunk i+1.

        **Validates: Requirements 3.2**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        if overlap == 0 or len(chunks) <= 1:
            return  # No overlap to verify

        for i in range(len(chunks) - 1):
            chunk_i_lines = chunks[i].text.split("\n")
            chunk_next_lines = chunks[i + 1].text.split("\n")

            # The last O messages of chunk i should equal first O messages of chunk i+1
            last_o_of_i = chunk_i_lines[-overlap:]
            first_o_of_next = chunk_next_lines[:overlap]

            assert last_o_of_i == first_o_of_next, (
                f"Overlap mismatch between chunk {i} and {i+1}: "
                f"last {overlap} of chunk {i} = {last_o_of_i}, "
                f"first {overlap} of chunk {i+1} = {first_o_of_next}"
            )

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_union_of_chunks_covers_all_messages(self, conversation, config):
        """(d) The union of all chunks covers all N messages.

        **Validates: Requirements 3.1, 3.2**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        # Collect all message texts found in chunks
        all_texts_in_chunks = set()
        for chunk in chunks:
            for line in chunk.text.split("\n"):
                all_texts_in_chunks.add(line)

        # Every original message must appear in at least one chunk
        for msg in conversation.messages:
            assert (
                msg.text in all_texts_in_chunks
            ), f"Message '{msg.text[:50]}...' not found in any chunk"


# --- Property 6: Chunk Metadata Completeness ---


class TestChunkMetadataCompleteness:
    """Property 6: Chunk Metadata Completeness.

    # Feature: sms-rag, Property 6: Chunk Metadata Completeness
    """

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_chunk_has_non_empty_participant_name(self, conversation, config):
        """Each chunk SHALL have a non-empty participant_name field.

        **Validates: Requirements 3.3**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            assert (
                chunk.participant_name
            ), f"Chunk {chunk.chunk_id} has empty participant_name"
            assert chunk.participant_name == conversation.participant_name

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_chunk_has_non_empty_source_filename(self, conversation, config):
        """Each chunk SHALL have a non-empty source_filename field.

        **Validates: Requirements 3.3**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            assert (
                chunk.source_filename
            ), f"Chunk {chunk.chunk_id} has empty source_filename"
            assert chunk.source_filename == conversation.source_filename

    @given(
        conversation=conversation_generator(min_messages=1, max_messages=80),
        config=chunk_config_generator(),
    )
    @settings(max_examples=25)
    def test_date_range_spans_constituent_message_timestamps(
        self, conversation, config
    ):
        """date_range_start/date_range_end SHALL span the timestamps of the
        constituent messages (when timestamps are present).

        **Validates: Requirements 3.3**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        # Reconstruct which messages belong to each chunk by matching text lines
        step = chunk_size - overlap
        messages = conversation.messages

        start = 0
        chunk_idx = 0
        while start < len(messages) and chunk_idx < len(chunks):
            end = min(start + chunk_size, len(messages))
            chunk_messages = messages[start:end]
            chunk = chunks[chunk_idx]

            # Get timestamps from the constituent messages
            timestamps = [
                msg.timestamp for msg in chunk_messages if msg.timestamp is not None
            ]

            if timestamps:
                expected_start = min(timestamps)
                expected_end = max(timestamps)
                assert chunk.date_range_start == expected_start, (
                    f"Chunk {chunk_idx}: date_range_start={chunk.date_range_start}, "
                    f"expected={expected_start}"
                )
                assert chunk.date_range_end == expected_end, (
                    f"Chunk {chunk_idx}: date_range_end={chunk.date_range_end}, "
                    f"expected={expected_end}"
                )
                # date_range_start <= date_range_end
                assert chunk.date_range_start <= chunk.date_range_end
            else:
                # No timestamps means both should be None
                assert (
                    chunk.date_range_start is None
                ), f"Chunk {chunk_idx}: expected None date_range_start when no timestamps"
                assert (
                    chunk.date_range_end is None
                ), f"Chunk {chunk_idx}: expected None date_range_end when no timestamps"

            chunk_idx += 1
            next_start = start + step
            if next_start >= len(messages) or end >= len(messages):
                break
            start = next_start
