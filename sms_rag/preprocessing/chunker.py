"""Conversation chunker that splits parsed conversations into overlapping chunks."""

import hashlib

from sms_rag.shared.models import ConversationChunk, Message, ParsedConversation


class ConversationChunker:
    """Groups messages into overlapping chunks suitable for embedding.

    Chunks never split a single message across boundaries. Adjacent chunks
    overlap by a configurable number of messages to preserve conversational
    context across chunk boundaries.
    """

    def __init__(self, chunk_size: int = 20, overlap: int = 5):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum number of messages per chunk.
            overlap: Number of messages shared between adjacent chunks.

        Raises:
            ValueError: If overlap >= chunk_size or either value is < 1.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, conversation: ParsedConversation) -> list[ConversationChunk]:
        """Split a parsed conversation into overlapping chunks.

        Args:
            conversation: A parsed conversation containing messages and metadata.

        Returns:
            A list of ConversationChunk objects. Conversations with no messages
            return an empty list. Conversations smaller than chunk_size produce
            a single chunk.
        """
        messages = conversation.messages
        if not messages:
            return []

        filename_hash = self._hash_filename(conversation.source_filename)
        step = self.chunk_size - self.overlap
        chunks: list[ConversationChunk] = []
        chunk_index = 0

        start = 0
        while start < len(messages):
            end = min(start + self.chunk_size, len(messages))
            chunk_messages = messages[start:end]

            chunk = self._build_chunk(
                messages=chunk_messages,
                chunk_id=f"{filename_hash}_{chunk_index}",
                participant_name=conversation.participant_name,
                source_filename=conversation.source_filename,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Move start forward by step; stop if we've already reached the end
            next_start = start + step
            if next_start >= len(messages):
                break
            # If next window would start past where we already covered everything
            if end >= len(messages):
                break
            start = next_start

        return chunks

    def _build_chunk(
        self,
        messages: list[Message],
        chunk_id: str,
        participant_name: str,
        source_filename: str,
    ) -> ConversationChunk:
        """Build a ConversationChunk from a list of messages with aggregated metadata."""
        text = "\n".join(msg.text for msg in messages)

        # Aggregate timestamps for date range
        timestamps = [msg.timestamp for msg in messages if msg.timestamp is not None]
        date_range_start = min(timestamps) if timestamps else None
        date_range_end = max(timestamps) if timestamps else None

        # Aggregate message types (deduplicated)
        message_types = list(
            {msg.message_type for msg in messages if msg.message_type is not None}
        )

        # Aggregate phone numbers (deduplicated)
        phone_numbers = list(
            {msg.phone_number for msg in messages if msg.phone_number is not None}
        )

        return ConversationChunk(
            chunk_id=chunk_id,
            text=text,
            participant_name=participant_name,
            source_filename=source_filename,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            message_types=message_types,
            phone_numbers=phone_numbers,
            message_count=len(messages),
        )

    @staticmethod
    def _hash_filename(filename: str) -> str:
        """Generate a deterministic hash from a filename."""
        return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
