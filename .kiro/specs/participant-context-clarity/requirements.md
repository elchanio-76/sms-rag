# Requirements Document

## Introduction

The RAG system currently presents retrieved conversation passages to the LLM in a flat, interleaved format ordered by relevance score. Each passage has only a brief parenthetical header identifying the participant. Within each passage, individual messages are concatenated without any indication of who sent versus who received them. This causes the LLM (llama3.1:8b via Ollama) to confuse separate user-participant conversations as a multi-party conversation between those participants. This feature improves context presentation through two complementary approaches: grouping passages by participant with explicit structural headers, and investigating speaker role tagging to distinguish sent from received messages within each chunk.

## Glossary

- **Orchestrator**: The RAGOrchestrator class in `sms_rag/query/orchestrator.py` that formats context passages and calls the LLM provider to generate responses.
- **Chunker**: The ConversationChunker class in `sms_rag/preprocessing/chunker.py` that groups parsed messages into overlapping ConversationChunk objects for embedding and retrieval.
- **PDF_Parser**: The PDFParser class in `sms_rag/preprocessing/pdf_parser.py` that extracts Message objects from conversation PDF exports using PyMuPDF.
- **Context_Formatter**: The component within the Orchestrator responsible for transforming retrieved SearchResult objects into a formatted context string for the LLM prompt.
- **Participant_Name**: The name of the person the user had a private conversation with, derived from the PDF filename stem.
- **Speaker_Role**: An indicator of whether a message was sent by the user or received from the participant.
- **Delivery_Receipt**: The Greek text "Παραδόθηκε" that appears in PDF exports after messages sent by the user, acting as a boundary marker.
- **Chunk_Text**: The concatenated text content of all messages within a ConversationChunk, stored in the `text` field.
- **Message**: A single extracted text unit from the PDF, represented by the Message dataclass with text, timestamp, message_type, and phone_number fields.
- **SearchResult**: A retrieval result containing chunk_id, text, metadata (including participant_name and date ranges), and a relevance score.

## Requirements

### Requirement 1: Group Context Passages by Participant

**User Story:** As a user querying the RAG system, I want retrieved conversation passages grouped by participant with clear structural headers, so that the LLM understands each group is a separate private conversation between me and that specific participant.

#### Acceptance Criteria

1. WHEN formatting context for the LLM prompt, THE Context_Formatter SHALL group all retrieved passages belonging to the same Participant_Name under a single section header.
2. THE Context_Formatter SHALL prefix each participant section with a markdown header in the format `## Conversation with {Participant_Name} (Your private conversation)`.
3. WHILE multiple participants have relevant passages, THE Context_Formatter SHALL present each participant section as a distinct block separated by a markdown horizontal rule (`---`) on its own line between sections.
4. WHEN passages from the same participant span different date ranges, THE Context_Formatter SHALL order passages within that participant section in ascending chronological order by date_range_start.
5. WHEN a single participant has multiple retrieved passages, THE Context_Formatter SHALL annotate each individual passage with a date range label in the format `[YYYY-MM-DD to YYYY-MM-DD]` derived from the passage metadata date_range_start and date_range_end.
6. THE Context_Formatter SHALL include all retrieved passages in the grouped output without discarding any results due to the regrouping.
7. IF a passage has no date_range_start metadata, THEN THE Context_Formatter SHALL place that passage after all date-annotated passages within the same participant section and omit the date range label for that passage.
8. WHILE multiple participants have relevant passages, THE Context_Formatter SHALL order participant sections by the highest retrieval score among passages in each section, in descending order.

### Requirement 2: Investigate Speaker Direction via Text Coordinates

**User Story:** As a developer, I want to investigate whether PyMuPDF text extraction coordinates can distinguish left-aligned (received) from right-aligned (sent) chat bubbles, so that speaker role tagging can be implemented if feasible.

#### Acceptance Criteria

1. WHEN extracting text from a PDF page, THE PDF_Parser SHALL extract the x-coordinate position (in points, from the left edge of the page) of each text block alongside the text content.
2. WHEN text blocks from a page are extracted with coordinates, THE PDF_Parser SHALL classify blocks into left-aligned and right-aligned groups based on a configurable x-position threshold expressed in points, defaulting to 50% of the page width.
3. IF fewer than 80% of the message text blocks on a page fall clearly into one of the two groups (left-aligned below threshold, right-aligned at or above threshold), THEN THE PDF_Parser SHALL log a warning indicating the page number and percentage of unclassified blocks, and fall back to the Delivery_Receipt heuristic for speaker attribution for that PDF.
4. IF 80% or more of the message text blocks across all pages of a PDF are consistently classified into left-aligned and right-aligned groups, THEN THE PDF_Parser SHALL assign a Speaker_Role of "sent" to right-aligned messages and "received" to left-aligned messages.
5. IF a text block's x-coordinate falls within 10 points of the configured threshold, THEN THE PDF_Parser SHALL treat that block as ambiguous and exclude it from the reliability percentage calculation in criterion 3.

### Requirement 3: Infer Speaker Direction via Delivery Receipt Heuristic

**User Story:** As a developer, I want to use the "Παραδόθηκε" delivery receipt pattern as a fallback heuristic for tagging sent messages, so that speaker roles can be determined even when coordinate-based detection is unreliable.

#### Acceptance Criteria

1. WHEN a Delivery_Receipt line is detected, THE PDF_Parser SHALL tag the last message accumulated before that receipt line (i.e., the message whose text block immediately precedes the receipt in document order) with Speaker_Role "sent".
2. IF no Delivery_Receipt line follows a message in document order and no coordinate-based classification is available for that message, THEN THE PDF_Parser SHALL assign Speaker_Role "unknown" to that message.
3. THE PDF_Parser SHALL never assign Speaker_Role "sent" to a message unless that message is immediately followed by a Delivery_Receipt line in document order or coordinate-based classification indicates "sent".
4. IF both coordinate-based classification and Delivery_Receipt heuristic produce a Speaker_Role for the same message, THEN THE PDF_Parser SHALL use the coordinate-based result and discard the heuristic result.
5. IF a Delivery_Receipt line is detected but no message has been accumulated preceding it, THEN THE PDF_Parser SHALL ignore that receipt line without assigning any Speaker_Role.
6. WHEN the PDF_Parser produces a Message object, THE PDF_Parser SHALL populate the Speaker_Role field with exactly one of the values: "sent", "received", or "unknown".

### Requirement 4: Extend Message Model with Speaker Direction

**User Story:** As a developer, I want the Message data model to include a speaker direction field, so that downstream components can use this information for formatting and context presentation.

#### Acceptance Criteria

1. THE Message dataclass SHALL include a `speaker_role` field of type `str` with allowed values limited to "sent", "received", or "unknown".
2. THE Message dataclass SHALL default the `speaker_role` field to "unknown" when the PDF parser cannot determine the direction of a message.
3. WHEN a Message has a `speaker_role` value of "sent" or "received", THE Chunker SHALL prepend the speaker role as a prefix in the format `[You]: ` or `[{Participant_Name}]: ` before the message text in the chunk text for that message.
4. IF a Message has a `speaker_role` value of "unknown", THEN THE Chunker SHALL prefix the message text with `[Message]: ` in the chunk text.

### Requirement 5: Format Speaker Roles in Chunk Text

**User Story:** As a user, I want each message within a conversation chunk to be prefixed with a speaker indicator, so that the LLM can distinguish what I said from what the other participant said.

#### Acceptance Criteria

1. WHEN a Message has Speaker_Role "sent", THE Chunker SHALL prefix that message text with `[You]: ` followed by the message text in the Chunk_Text.
2. WHEN a Message has Speaker_Role "received", THE Chunker SHALL prefix that message text with `[{Participant_Name}]: ` in the Chunk_Text, where `{Participant_Name}` is replaced by the conversation's participant_name value.
3. WHEN a Message has Speaker_Role "unknown", THE Chunker SHALL prefix that message text with `[Message]: ` in the Chunk_Text.
4. THE Chunker SHALL separate each prefixed message with a single newline character (`\n`) in the Chunk_Text.
5. IF a message text contains newline characters, THEN THE Chunker SHALL preserve them within the prefixed message such that only the newline characters inserted by criterion 4 serve as message boundaries.
6. THE Chunker SHALL produce Chunk_Text such that splitting on the pattern `\n` followed by a prefix marker (`[You]: `, `[{Participant_Name}]: `, or `[Message]: `) recovers the original message texts exactly (round-trip property).

### Requirement 6: Maintain Backward Compatibility

**User Story:** As a user of the existing RAG system, I want the new formatting changes to work without breaking existing queries or requiring re-indexing of unchanged data, so that the system remains functional during incremental improvements.

#### Acceptance Criteria

1. WHEN a retrieved SearchResult's text field does not contain any speaker role prefix patterns (`[You]:`, `[{Participant_Name}]:`, or `[Message]:`), THE Context_Formatter SHALL present that chunk's text verbatim without injecting or stripping any prefix characters.
2. THE Orchestrator SHALL produce a valid GenerationResult with a non-empty `text` field and a non-empty `source_chunks` list regardless of whether the retrieved context passages contain speaker role prefixes, lack speaker role prefixes, or contain a mix of both.
3. WHEN re-indexing is performed with the updated Chunker on a PDF where at least one Message has a resolved Speaker_Role of "sent" or "received", THE Chunker SHALL produce chunks where every message line begins with one of the prefixes `[You]:`, `[{Participant_Name}]:`, or `[Message]:`.
4. IF the coordinate-based and receipt-based speaker detection both fail for an entire PDF (all messages have Speaker_Role "unknown"), THEN THE Chunker SHALL still produce valid ConversationChunk objects where every message line is prefixed with `[Message]:` and each chunk has a non-empty `text` field, a non-empty `participant_name`, and a `message_count` greater than zero.
5. WHEN the system processes a query against an index containing a mix of old chunks (without speaker prefixes) and new chunks (with speaker prefixes), THE Orchestrator SHALL format and pass both types to the LLM without raising an error or discarding either type.

### Requirement 7: Update System Prompt for Grouped Context

**User Story:** As a developer, I want the system prompt to instruct the LLM about the grouped context format, so that the model correctly interprets participant sections as separate private conversations.

#### Acceptance Criteria

1. WHEN the Orchestrator constructs the prompt for the LLM provider, THE Orchestrator SHALL include in the system prompt a statement that each `## Conversation with ...` section represents a separate private conversation between the user and that named participant.
2. THE Orchestrator SHALL include in the system prompt a statement that participants did not converse with each other and that all messages within a section are exclusively between the user and the participant named in that section header.
3. WHEN the formatted context contains at least one line matching the `[You]:` or `[{Name}]:` speaker role prefix pattern, THE Orchestrator SHALL include in the system prompt a statement that `[You]:` indicates messages sent by the user and `[{Name}]:` indicates messages received from the participant named in the enclosing `## Conversation with {Name}` section header.
4. IF the retrieved context contains chunks from more than one participant, THEN THE Orchestrator SHALL format the context with separate `## Conversation with {Name}` sections, one per participant, grouping all chunks for the same participant under their respective section.
