"""Data models shared across preprocessing and query pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Message:
    """A single message extracted from a conversation PDF."""

    text: str
    timestamp: datetime | None = None
    message_type: str | None = None  # "SMS", "iMessage", "RCS", or None
    phone_number: str | None = None
    speaker_role: str = "unknown"  # "sent", "received", or "unknown"


@dataclass
class TextBlock:
    """A text block extracted from a PDF page with spatial information."""

    text: str
    x_position: float  # X-coordinate in points from left edge
    page_number: int


@dataclass
class ParsedConversation:
    """The complete extraction result for one PDF file."""

    participant_name: str
    source_filename: str
    messages: list[Message] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ConversationChunk:
    """A chunk of conversation suitable for embedding and retrieval."""

    chunk_id: str
    text: str
    participant_name: str
    source_filename: str
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    message_types: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    message_count: int = 0


@dataclass
class SearchResult:
    """A single result from a similarity search query."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Normalized 0.0–1.0


@dataclass
class GenerationResult:
    """The result of an LLM generation with source references."""

    text: str
    source_chunks: list[SearchResult] = field(default_factory=list)


@dataclass
class PipelineSummary:
    """Summary of a preprocessing pipeline run."""

    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    files_errored: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SessionExchange:
    """A single user-assistant exchange in the session history."""

    exchange_id: str
    user_message: str
    assistant_response: str
    source_references: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
