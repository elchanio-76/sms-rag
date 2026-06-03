"""Unit tests for the ConversationChunker."""

from datetime import datetime

import pytest

from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.shared.models import ConversationChunk, Message, ParsedConversation


class TestConversationChunkerInit:
    """Tests for ConversationChunker initialization and validation."""

    def test_default_parameters(self):
        chunker = ConversationChunker()
        assert chunker.chunk_size == 20
        assert chunker.overlap == 5

    def test_custom_parameters(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        assert chunker.chunk_size == 10
        assert chunker.overlap == 2

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            ConversationChunker(chunk_size=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            ConversationChunker(overlap=-1)

    def test_overlap_greater_than_or_equal_chunk_size(self):
        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ConversationChunker(chunk_size=5, overlap=5)

        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ConversationChunker(chunk_size=5, overlap=10)


class TestConversationChunkerChunk:
    """Tests for the chunk() method."""

    def _make_conversation(
        self, n_messages: int, filename: str = "Test User.pdf"
    ) -> ParsedConversation:
        """Helper to create a conversation with numbered messages."""
        messages = [
            Message(
                text=f"Message {i}",
                timestamp=datetime(2024, 1, 1, i % 24, 0),
                message_type="SMS" if i % 2 == 0 else "iMessage",
                phone_number=f"+1234567890{i % 3}",
            )
            for i in range(n_messages)
        ]
        return ParsedConversation(
            participant_name="Test User",
            source_filename=filename,
            messages=messages,
        )

    def test_empty_conversation(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = ParsedConversation(
            participant_name="Test", source_filename="Test.pdf", messages=[]
        )
        result = chunker.chunk(conv)
        assert result == []

    def test_single_chunk_when_fewer_messages_than_chunk_size(self):
        chunker = ConversationChunker(chunk_size=20, overlap=5)
        conv = self._make_conversation(10)
        result = chunker.chunk(conv)
        assert len(result) == 1
        assert result[0].message_count == 10

    def test_single_chunk_when_exactly_chunk_size(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(5)
        result = chunker.chunk(conv)
        assert len(result) == 1
        assert result[0].message_count == 5

    def test_overlapping_chunks(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(8)
        result = chunker.chunk(conv)

        # With 8 messages, chunk_size=5, overlap=2: step=3
        # Chunk 0: messages[0:5] (5 messages)
        # Chunk 1: messages[3:8] (5 messages)
        assert len(result) == 2
        assert result[0].message_count == 5
        assert result[1].message_count == 5

        # Verify overlap: last 2 of chunk 0 = first 2 of chunk 1
        chunk0_texts = result[0].text.split("\n")
        chunk1_texts = result[1].text.split("\n")
        assert chunk0_texts[-2:] == chunk1_texts[:2]

    def test_three_chunks_with_overlap(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(11)
        result = chunker.chunk(conv)

        # step=3, chunks: [0:5], [3:8], [6:11]
        assert len(result) == 3
        assert result[0].message_count == 5
        assert result[1].message_count == 5
        assert result[2].message_count == 5

    def test_deterministic_chunk_ids(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(8)

        result1 = chunker.chunk(conv)
        result2 = chunker.chunk(conv)

        assert result1[0].chunk_id == result2[0].chunk_id
        assert result1[1].chunk_id == result2[1].chunk_id

    def test_chunk_id_format(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(8, filename="MyFile.pdf")
        result = chunker.chunk(conv)

        for i, chunk in enumerate(result):
            parts = chunk.chunk_id.split("_")
            assert len(parts) == 2
            assert parts[1] == str(i)
            # Hash should be a hex string
            assert all(c in "0123456789abcdef" for c in parts[0])

    def test_different_filenames_produce_different_ids(self):
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv1 = self._make_conversation(8, filename="Alice.pdf")
        conv2 = self._make_conversation(8, filename="Bob.pdf")

        result1 = chunker.chunk(conv1)
        result2 = chunker.chunk(conv2)

        assert result1[0].chunk_id != result2[0].chunk_id

    def test_text_is_newline_joined(self):
        chunker = ConversationChunker(chunk_size=3, overlap=1)
        messages = [
            Message(text="Hello"),
            Message(text="World"),
            Message(text="Foo"),
        ]
        conv = ParsedConversation(
            participant_name="Test",
            source_filename="Test.pdf",
            messages=messages,
        )
        result = chunker.chunk(conv)
        assert result[0].text == "Hello\nWorld\nFoo"

    def test_metadata_aggregation_participant_and_filename(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        conv = self._make_conversation(5, filename="Kyriaki Salavanitou.pdf")
        conv.participant_name = "Kyriaki Salavanitou"
        result = chunker.chunk(conv)

        assert result[0].participant_name == "Kyriaki Salavanitou"
        assert result[0].source_filename == "Kyriaki Salavanitou.pdf"

    def test_metadata_aggregation_date_range(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        messages = [
            Message(text="First", timestamp=datetime(2024, 1, 1, 10, 0)),
            Message(text="Second", timestamp=datetime(2024, 1, 2, 12, 0)),
            Message(text="Third", timestamp=datetime(2024, 1, 3, 8, 0)),
        ]
        conv = ParsedConversation(
            participant_name="Test",
            source_filename="Test.pdf",
            messages=messages,
        )
        result = chunker.chunk(conv)

        assert result[0].date_range_start == datetime(2024, 1, 1, 10, 0)
        assert result[0].date_range_end == datetime(2024, 1, 3, 8, 0)

    def test_metadata_aggregation_no_timestamps(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        messages = [
            Message(text="First"),
            Message(text="Second"),
        ]
        conv = ParsedConversation(
            participant_name="Test",
            source_filename="Test.pdf",
            messages=messages,
        )
        result = chunker.chunk(conv)

        assert result[0].date_range_start is None
        assert result[0].date_range_end is None

    def test_metadata_aggregation_message_types_deduplicated(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        messages = [
            Message(text="A", message_type="SMS"),
            Message(text="B", message_type="SMS"),
            Message(text="C", message_type="iMessage"),
            Message(text="D", message_type=None),
        ]
        conv = ParsedConversation(
            participant_name="Test",
            source_filename="Test.pdf",
            messages=messages,
        )
        result = chunker.chunk(conv)

        assert sorted(result[0].message_types) == ["SMS", "iMessage"]

    def test_metadata_aggregation_phone_numbers_deduplicated(self):
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        messages = [
            Message(text="A", phone_number="+123"),
            Message(text="B", phone_number="+123"),
            Message(text="C", phone_number="+456"),
            Message(text="D", phone_number=None),
        ]
        conv = ParsedConversation(
            participant_name="Test",
            source_filename="Test.pdf",
            messages=messages,
        )
        result = chunker.chunk(conv)

        assert sorted(result[0].phone_numbers) == ["+123", "+456"]

    def test_zero_overlap(self):
        chunker = ConversationChunker(chunk_size=3, overlap=0)
        conv = self._make_conversation(7)
        result = chunker.chunk(conv)

        # step=3, chunks: [0:3], [3:6], [6:7]
        assert len(result) == 3
        assert result[0].message_count == 3
        assert result[1].message_count == 3
        assert result[2].message_count == 1

    def test_all_messages_covered(self):
        """Verify that the union of all chunks covers all messages."""
        chunker = ConversationChunker(chunk_size=5, overlap=2)
        conv = self._make_conversation(13)
        result = chunker.chunk(conv)

        # Collect all message texts from all chunks
        all_texts_in_chunks: set[str] = set()
        for chunk in result:
            for line in chunk.text.split("\n"):
                all_texts_in_chunks.add(line)

        # All original messages should be present
        for msg in conv.messages:
            assert msg.text in all_texts_in_chunks
