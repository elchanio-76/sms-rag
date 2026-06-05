"""Property-based tests for the ConversationChunker.

# Feature: sms-rag, Property 5: Chunking Structural Correctness
# Feature: sms-rag, Property 6: Chunk Metadata Completeness
# Feature: participant-context-clarity, Property 14: Speaker Role Prefix Round-Trip
# Feature: participant-context-clarity, Property 15: Speaker Role Prefix Correctness

Validates: Requirements 3.1, 3.2, 3.3, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6

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

import re
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
            # Each line should be a complete message text from the conversation (with speaker prefix)
            for line in lines:
                assert any(
                    line.endswith(msg.text)
                    and (
                        line == f"[You]: {msg.text}"
                        or line == f"[{conversation.participant_name}]: {msg.text}"
                        or line == f"[Message]: {msg.text}"
                    )
                    for msg in conversation.messages
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

        # Collect all message texts found in chunks (with speaker prefixes)
        all_texts_in_chunks = set()
        for chunk in chunks:
            for line in chunk.text.split("\n"):
                all_texts_in_chunks.add(line)

        # Every original message must appear in at least one chunk (with prefix)
        for msg in conversation.messages:
            prefixed = ConversationChunker._format_message_line(
                msg, conversation.participant_name
            )
            assert (
                prefixed in all_texts_in_chunks
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


# --- Generators for Property 15 ---


@st.composite
def message_with_role_generator(draw):
    """Generate a random Message object with a speaker_role from {sent, received, unknown}."""
    text = draw(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_characters="\x00\n",
            ),
        )
    )

    speaker_role = draw(st.sampled_from(["sent", "received", "unknown"]))

    return Message(
        text=text,
        speaker_role=speaker_role,
    )


@st.composite
def participant_name_generator(draw):
    """Generate a random participant name string."""
    return draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00\n",
            ),
        )
    )


# --- Property 15: Speaker Role Prefix Correctness ---


class TestSpeakerRolePrefixCorrectness:
    """Property 15: Speaker Role Prefix Correctness.

    # Feature: participant-context-clarity, Property 15: Speaker Role Prefix Correctness

    For any Message with speaker_role "sent", the corresponding line in the
    chunk text SHALL begin with `[You]: `. For "received" with participant
    name P, the line SHALL begin with `[{P}]: `. For "unknown", the line
    SHALL begin with `[Message]: `.
    """

    @given(
        message=message_with_role_generator(),
        participant_name=participant_name_generator(),
    )
    @settings(max_examples=100)
    def test_sent_message_has_you_prefix(self, message, participant_name):
        """Messages with speaker_role 'sent' SHALL begin with '[You]: '.

        **Validates: Requirements 4.3, 5.1**
        """
        if message.speaker_role != "sent":
            return  # Only test sent messages

        result = ConversationChunker._format_message_line(message, participant_name)

        assert result.startswith("[You]: "), (
            f"Expected line to start with '[You]: ' for sent message, "
            f"got: '{result[:30]}...'"
        )
        assert result == f"[You]: {message.text}"

    @given(
        message=message_with_role_generator(),
        participant_name=participant_name_generator(),
    )
    @settings(max_examples=100)
    def test_received_message_has_participant_prefix(self, message, participant_name):
        """Messages with speaker_role 'received' SHALL begin with '[{participant_name}]: '.

        **Validates: Requirements 4.3, 5.2**
        """
        if message.speaker_role != "received":
            return  # Only test received messages

        result = ConversationChunker._format_message_line(message, participant_name)

        expected_prefix = f"[{participant_name}]: "
        assert result.startswith(expected_prefix), (
            f"Expected line to start with '{expected_prefix}' for received message, "
            f"got: '{result[:50]}...'"
        )
        assert result == f"[{participant_name}]: {message.text}"

    @given(
        message=message_with_role_generator(),
        participant_name=participant_name_generator(),
    )
    @settings(max_examples=100)
    def test_unknown_message_has_message_prefix(self, message, participant_name):
        """Messages with speaker_role 'unknown' SHALL begin with '[Message]: '.

        **Validates: Requirements 4.4, 5.3**
        """
        if message.speaker_role != "unknown":
            return  # Only test unknown messages

        result = ConversationChunker._format_message_line(message, participant_name)

        assert result.startswith("[Message]: "), (
            f"Expected line to start with '[Message]: ' for unknown message, "
            f"got: '{result[:30]}...'"
        )
        assert result == f"[Message]: {message.text}"

    @given(
        messages=st.lists(message_with_role_generator(), min_size=1, max_size=20),
        participant_name=participant_name_generator(),
    )
    @settings(max_examples=100)
    def test_all_roles_have_correct_prefix_in_chunk(self, messages, participant_name):
        """When messages are chunked together, each line starts with the correct prefix.

        **Validates: Requirements 4.3, 4.4, 5.1, 5.2, 5.3**
        """
        # Build chunk text using the chunker's method
        text = "\n".join(
            ConversationChunker._format_message_line(msg, participant_name)
            for msg in messages
        )
        lines = text.split("\n")

        assert len(lines) == len(messages)

        for line, msg in zip(lines, messages):
            if msg.speaker_role == "sent":
                assert line.startswith(
                    "[You]: "
                ), f"Sent message line should start with '[You]: ', got: '{line[:30]}'"
                assert line == f"[You]: {msg.text}"
            elif msg.speaker_role == "received":
                expected_prefix = f"[{participant_name}]: "
                assert line.startswith(expected_prefix), (
                    f"Received message line should start with '{expected_prefix}', "
                    f"got: '{line[:50]}'"
                )
                assert line == f"[{participant_name}]: {msg.text}"
            else:
                assert line.startswith("[Message]: "), (
                    f"Unknown message line should start with '[Message]: ', "
                    f"got: '{line[:30]}'"
                )
                assert line == f"[Message]: {msg.text}"


# --- Generators for Property 14 ---


@st.composite
def message_with_role_and_newlines_generator(draw):
    """Generate a Message with speaker_role that may contain internal newlines.

    This generator allows newline characters within the message text to validate
    that the round-trip property holds even when messages contain embedded newlines
    (Requirement 5.5).
    """
    # Generate text segments (at least one non-empty segment)
    segments = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                    blacklist_characters="\x00",
                ),
            ),
            min_size=1,
            max_size=3,
        )
    )

    # Join segments with optional internal newlines
    use_newlines = draw(st.booleans())
    if use_newlines and len(segments) > 1:
        text = "\n".join(segments)
    else:
        text = " ".join(segments)

    speaker_role = draw(st.sampled_from(["sent", "received", "unknown"]))

    return Message(text=text, speaker_role=speaker_role)


@st.composite
def participant_name_no_brackets_generator(draw):
    """Generate a participant name that does not contain ']' to avoid ambiguity in prefix parsing.

    The round-trip regex `\\n(?=\\[(?:You|Message|[^\\]]+)\\]: )` relies on the fact that
    participant names don't contain ']'. This is a reasonable real-world constraint.
    """
    return draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00\n]",
            ),
        )
    )


# --- Property 14: Speaker Role Prefix Round-Trip ---


# The regex pattern for splitting chunk text back into individual prefixed messages.
# Splits on a newline that is immediately followed by a speaker prefix pattern.
_ROUND_TRIP_SPLIT_PATTERN = re.compile(r"\n(?=\[(?:You|Message|[^\]]+)\]: )")

# The regex pattern for stripping the prefix from a message line.
_PREFIX_STRIP_PATTERN = re.compile(r"^\[(?:You|Message|[^\]]+)\]: ")


class TestSpeakerRolePrefixRoundTrip:
    """Property 14: Speaker Role Prefix Round-Trip.

    # Feature: participant-context-clarity, Property 14: Speaker Role Prefix Round-Trip

    For any list of Message objects with known speaker_roles and a participant name,
    chunking the messages into a ConversationChunk and then splitting the chunk text
    on the pattern `\\n(?=\\[(?:You|Message|[^\\]]+)\\]: )` SHALL recover the original
    message texts exactly (after stripping the prefix from each segment).

    **Validates: Requirements 5.4, 5.5, 5.6**
    """

    @given(
        messages=st.lists(message_with_role_generator(), min_size=1, max_size=20),
        participant_name=participant_name_no_brackets_generator(),
    )
    @settings(max_examples=100)
    def test_round_trip_without_internal_newlines(self, messages, participant_name):
        """Messages without internal newlines round-trip exactly through chunk text.

        **Validates: Requirements 5.4, 5.6**
        """
        # Build chunk text using the chunker's method
        chunk_text = "\n".join(
            ConversationChunker._format_message_line(msg, participant_name)
            for msg in messages
        )

        # Split using the round-trip regex
        recovered_lines = _ROUND_TRIP_SPLIT_PATTERN.split(chunk_text)

        assert len(recovered_lines) == len(
            messages
        ), f"Expected {len(messages)} segments after split, got {len(recovered_lines)}"

        # Strip prefixes and compare to original texts
        for recovered_line, original_msg in zip(recovered_lines, messages):
            stripped = _PREFIX_STRIP_PATTERN.sub("", recovered_line, count=1)
            assert (
                stripped == original_msg.text
            ), f"Round-trip mismatch: recovered '{stripped}' != original '{original_msg.text}'"

    @given(
        messages=st.lists(
            message_with_role_and_newlines_generator(), min_size=1, max_size=10
        ),
        participant_name=participant_name_no_brackets_generator(),
    )
    @settings(max_examples=100)
    def test_round_trip_with_internal_newlines(self, messages, participant_name):
        """Messages with internal newlines round-trip exactly through chunk text.

        Internal newlines within a message are preserved and only the newlines
        inserted by the chunker as message separators serve as boundaries.

        **Validates: Requirements 5.4, 5.5, 5.6**
        """
        # Build chunk text using the chunker's method
        chunk_text = "\n".join(
            ConversationChunker._format_message_line(msg, participant_name)
            for msg in messages
        )

        # Split using the round-trip regex
        recovered_lines = _ROUND_TRIP_SPLIT_PATTERN.split(chunk_text)

        assert len(recovered_lines) == len(
            messages
        ), f"Expected {len(messages)} segments after split, got {len(recovered_lines)}"

        # Strip prefixes and compare to original texts
        for recovered_line, original_msg in zip(recovered_lines, messages):
            stripped = _PREFIX_STRIP_PATTERN.sub("", recovered_line, count=1)
            assert (
                stripped == original_msg.text
            ), f"Round-trip mismatch: recovered '{stripped}' != original '{original_msg.text}'"

    @given(
        messages=st.lists(
            message_with_role_and_newlines_generator(), min_size=1, max_size=15
        ),
        participant_name=participant_name_no_brackets_generator(),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_round_trip_through_full_chunker(self, messages, participant_name, config):
        """Round-trip property holds when using the full chunker pipeline.

        Messages chunked via ConversationChunker.chunk() can be recovered
        by splitting each chunk's text on the round-trip regex.

        **Validates: Requirements 5.4, 5.5, 5.6**
        """
        chunk_size, overlap = config

        # Create a ParsedConversation
        conversation = ParsedConversation(
            participant_name=participant_name,
            source_filename="test.pdf",
            messages=messages,
        )

        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        # For each chunk, verify round-trip
        for chunk in chunks:
            recovered_lines = _ROUND_TRIP_SPLIT_PATTERN.split(chunk.text)

            # Each recovered line should have a valid prefix and strip cleanly
            for recovered_line in recovered_lines:
                assert _PREFIX_STRIP_PATTERN.match(recovered_line), (
                    f"Recovered line doesn't start with a valid prefix: "
                    f"'{recovered_line[:50]}...'"
                )
                stripped = _PREFIX_STRIP_PATTERN.sub("", recovered_line, count=1)
                # The stripped text should be non-empty (since messages have min_size=1)
                assert (
                    len(stripped) > 0
                ), f"Stripped text is empty for line: '{recovered_line[:50]}...'"

            # Number of recovered segments should equal message_count
            assert len(recovered_lines) == chunk.message_count, (
                f"Chunk has message_count={chunk.message_count} but split produced "
                f"{len(recovered_lines)} segments"
            )


# --- Generators for Property 17 ---


@st.composite
def all_unknown_conversation_generator(draw, min_messages=1, max_messages=50):
    """Generate a ParsedConversation where ALL messages have speaker_role 'unknown'.

    This validates graceful degradation when both coordinate-based and
    receipt-based speaker detection fail for an entire PDF.
    """
    num_messages = draw(st.integers(min_value=min_messages, max_value=max_messages))

    messages = []
    for _ in range(num_messages):
        text = draw(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                    blacklist_characters="\x00\n",
                ),
            )
        )
        messages.append(Message(text=text, speaker_role="unknown"))

    participant_name = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00\n",
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


# --- Property 17: Full-Unknown Graceful Degradation ---


class TestFullUnknownGracefulDegradation:
    """Property 17: Full-Unknown Graceful Degradation.

    # Feature: participant-context-clarity, Property 17: Full-Unknown Graceful Degradation

    For any ParsedConversation where all messages have speaker_role "unknown",
    the Chunker SHALL produce valid ConversationChunk objects where every message
    line is prefixed with `[Message]:`, and each chunk has non-empty `text`,
    non-empty `participant_name`, and `message_count > 0`.

    **Validates: Requirements 6.3, 6.4**
    """

    @given(
        conversation=all_unknown_conversation_generator(
            min_messages=1, max_messages=50
        ),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_all_lines_prefixed_with_message(self, conversation, config):
        """All lines in chunks SHALL be prefixed with '[Message]: ' when all roles are unknown.

        **Validates: Requirements 6.3**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            lines = chunk.text.split("\n")
            for line in lines:
                assert line.startswith("[Message]: "), (
                    f"Expected all lines to start with '[Message]: ' "
                    f"when all roles are unknown, got: '{line[:50]}...'"
                )

    @given(
        conversation=all_unknown_conversation_generator(
            min_messages=1, max_messages=50
        ),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_chunks_have_non_empty_text(self, conversation, config):
        """Each chunk SHALL have non-empty text field.

        **Validates: Requirements 6.4**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        assert len(chunks) > 0, "Chunker should produce at least one chunk"

        for chunk in chunks:
            assert chunk.text, f"Chunk {chunk.chunk_id} has empty text field"
            assert (
                len(chunk.text.strip()) > 0
            ), f"Chunk {chunk.chunk_id} has whitespace-only text"

    @given(
        conversation=all_unknown_conversation_generator(
            min_messages=1, max_messages=50
        ),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_chunks_have_non_empty_participant_name(self, conversation, config):
        """Each chunk SHALL have non-empty participant_name.

        **Validates: Requirements 6.4**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        assert len(chunks) > 0, "Chunker should produce at least one chunk"

        for chunk in chunks:
            assert (
                chunk.participant_name
            ), f"Chunk {chunk.chunk_id} has empty participant_name"

    @given(
        conversation=all_unknown_conversation_generator(
            min_messages=1, max_messages=50
        ),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_chunks_have_positive_message_count(self, conversation, config):
        """Each chunk SHALL have message_count > 0.

        **Validates: Requirements 6.4**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        assert len(chunks) > 0, "Chunker should produce at least one chunk"

        for chunk in chunks:
            assert chunk.message_count > 0, (
                f"Chunk {chunk.chunk_id} has message_count={chunk.message_count}, "
                f"expected > 0"
            )

    @given(
        conversation=all_unknown_conversation_generator(
            min_messages=1, max_messages=50
        ),
        config=chunk_config_generator(),
    )
    @settings(max_examples=100)
    def test_message_count_matches_lines(self, conversation, config):
        """Each chunk's message_count SHALL match the number of prefixed lines in text.

        **Validates: Requirements 6.3, 6.4**
        """
        chunk_size, overlap = config
        chunker = ConversationChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(conversation)

        for chunk in chunks:
            lines = chunk.text.split("\n")
            assert len(lines) == chunk.message_count, (
                f"Chunk {chunk.chunk_id}: text has {len(lines)} lines but "
                f"message_count={chunk.message_count}"
            )
