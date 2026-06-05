# Implementation Plan: Participant Context Clarity

## Overview

This implementation improves RAG context presentation through two complementary tracks: (1) grouping retrieved passages by participant with structural headers in the orchestrator, and (2) investigating speaker role tagging via x-coordinate detection and delivery receipt heuristic in the parser/chunker. The plan proceeds model-first (extend Message), then parser (coordinate + receipt detection), then chunker (prefix formatting), then orchestrator (grouped context + system prompt), with property tests interleaved near their implementation.

## Tasks

- [x] 1. Extend Message model and add TextBlock dataclass
  - [x] 1.1 Add `speaker_role` field to Message dataclass and create TextBlock dataclass
    - Add `speaker_role: str = "unknown"` field to the existing `Message` dataclass in `sms_rag/shared/models.py`
    - Create a new `TextBlock` dataclass with fields: `text: str`, `x_position: float`, `page_number: int`
    - Ensure allowed values for `speaker_role` are documented as "sent", "received", or "unknown"
    - _Requirements: 4.1, 4.2_

  - [x] 1.2 Write property test for Speaker Role Domain Invariant
    - **Property 13: Speaker Role Domain Invariant**
    - **Validates: Requirements 3.6, 4.1, 4.2**
    - Create a generator `message_with_role_generator` that produces Message objects with arbitrary text and speaker_role values
    - Assert that every Message produced by the parser has speaker_role in {"sent", "received", "unknown"}
    - Place test in `tests/test_property_pdf_parser.py`

- [ ] 2. Implement coordinate-based speaker detection in PDF Parser
  - [x] 2.1 Add constructor parameters and `_extract_text_blocks_with_coords` method
    - Add `x_threshold_ratio: float = 0.5` and `ambiguity_margin: float = 10.0` to `PDFParser.__init__`
    - Implement `_extract_text_blocks_with_coords(self, page: pymupdf.Page) -> list[TextBlock]` using `page.get_text("dict")` to extract block-level x-coordinates
    - Each TextBlock captures the text content, the x0 coordinate (left edge), and page number
    - _Requirements: 2.1_

  - [x] 2.2 Implement `_classify_speaker_by_coordinates` method
    - Calculate threshold as `page_width * self._x_threshold_ratio`
    - For each text block, determine if within ambiguity margin (exclude from classification)
    - Classify non-ambiguous blocks: `x < threshold` → "received", `x >= threshold` → "sent"
    - If fewer than 80% of non-ambiguous blocks are classifiable → log warning, return None (triggers fallback)
    - If ≥80% classifiable → return dict mapping text → speaker_role
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [ ] 2.3 Write property test for Coordinate Classification Correctness
    - **Property 7: Coordinate Classification Correctness**
    - **Validates: Requirements 2.2, 2.4**
    - Use `text_block_generator(page_width, threshold_ratio, ambiguity_margin)` to generate TextBlock lists
    - Assert that non-ambiguous blocks below threshold get "received" and at/above threshold get "sent"
    - Place test in `tests/test_property_pdf_parser.py`

  - [ ] 2.4 Write property test for Ambiguity Exclusion
    - **Property 8: Ambiguity Exclusion**
    - **Validates: Requirements 2.5**
    - Generate text blocks with x-coordinates within `ambiguity_margin` of the threshold
    - Assert these blocks are excluded from the reliability percentage and don't receive coordinate-based roles
    - Place test in `tests/test_property_pdf_parser.py`

  - [ ] 2.5 Write property test for Coordinate Reliability Threshold
    - **Property 9: Coordinate Reliability Threshold**
    - **Validates: Requirements 2.3**
    - Generate block sets where <80% of non-ambiguous blocks are classifiable
    - Assert classification returns None (fallback triggered)
    - Generate block sets where ≥80% are classifiable and assert coordinate roles are returned
    - Place test in `tests/test_property_pdf_parser.py`

- [ ] 3. Implement delivery receipt heuristic in PDF Parser
  - [ ] 3.1 Implement `_assign_speaker_roles_by_receipt` method
    - Iterate messages in document order; when a delivery receipt ("Παραδόθηκε") is detected after a message, tag that message as "sent"
    - Messages without a following receipt and without coordinate classification get "unknown"
    - Orphan receipts (no preceding message) are ignored
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [ ] 3.2 Integrate coordinate and receipt detection into `_extract_messages`
    - Modify `parse()` to first attempt coordinate-based classification
    - If coordinate detection is reliable (≥80% threshold met), apply coordinate roles to all messages
    - If coordinate detection fails, apply delivery receipt heuristic as fallback
    - Coordinate-based classification takes precedence over receipt-based
    - Ensure every Message object gets exactly one of "sent", "received", or "unknown"
    - _Requirements: 3.4, 3.6, 2.3_

  - [ ] 3.3 Write property test for Delivery Receipt Tagging
    - **Property 10: Delivery Receipt Tagging**
    - **Validates: Requirements 3.1, 3.2**
    - Generate message sequences with delivery receipts in various positions
    - Assert messages followed by receipt get "sent", others get "unknown"
    - Place test in `tests/test_property_pdf_parser.py`

  - [ ] 3.4 Write property test for Speaker Role Assignment Safety Invariant
    - **Property 11: Speaker Role Assignment Safety Invariant**
    - **Validates: Requirements 3.3**
    - For any set of messages, assert that "sent" is only assigned when justified by receipt or coordinate classification
    - Place test in `tests/test_property_pdf_parser.py`

  - [ ] 3.5 Write property test for Coordinate Precedence Over Heuristic
    - **Property 12: Coordinate Precedence Over Heuristic**
    - **Validates: Requirements 3.4**
    - Generate messages where both methods produce roles; assert coordinate result wins
    - Place test in `tests/test_property_pdf_parser.py`

- [ ] 4. Checkpoint - Ensure all parser tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Update Chunker with speaker role prefixing
  - [ ] 5.1 Implement `_format_message_line` static method in ConversationChunker
    - Add `_format_message_line(message: Message, participant_name: str) -> str` method
    - Return `[You]: {text}` for "sent", `[{participant_name}]: {text}` for "received", `[Message]: {text}` for "unknown"
    - _Requirements: 4.3, 4.4, 5.1, 5.2, 5.3_

  - [ ] 5.2 Update `_build_chunk` to use speaker role prefixes
    - Replace `text = "\n".join(msg.text for msg in messages)` with `text = "\n".join(self._format_message_line(msg, participant_name) for msg in messages)`
    - Ensure messages are separated by single newline characters
    - Preserve internal newlines within message text
    - _Requirements: 5.4, 5.5_

  - [ ] 5.3 Write property test for Speaker Role Prefix Correctness
    - **Property 15: Speaker Role Prefix Correctness**
    - **Validates: Requirements 4.3, 4.4, 5.1, 5.2, 5.3**
    - Generate messages with various roles and participant names
    - Assert each line starts with the correct prefix based on speaker_role
    - Place test in `tests/test_property_chunker.py`

  - [ ] 5.4 Write property test for Speaker Role Prefix Round-Trip
    - **Property 14: Speaker Role Prefix Round-Trip**
    - **Validates: Requirements 5.4, 5.5, 5.6**
    - Generate messages, chunk them, split chunk text on `\n(?=\[(?:You|Message|[^\]]+)\]: )`, strip prefixes
    - Assert recovered texts match original message texts exactly
    - Place test in `tests/test_property_chunker.py`

  - [ ] 5.5 Write property test for Full-Unknown Graceful Degradation
    - **Property 17: Full-Unknown Graceful Degradation**
    - **Validates: Requirements 6.3, 6.4**
    - Generate ParsedConversation where all messages have "unknown" role
    - Assert chunks have all lines prefixed with `[Message]:`, non-empty text, non-empty participant_name, message_count > 0
    - Place test in `tests/test_property_chunker.py`

- [ ] 6. Checkpoint - Ensure all chunker tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Rewrite Context Formatter with participant grouping
  - [ ] 7.1 Implement participant-grouped `_format_context` and `_format_date_label` in RAGOrchestrator
    - Replace existing `_format_context` method with new implementation that groups chunks by `participant_name` metadata
    - Order participant sections by highest max score descending
    - Within each section: sort dated passages ascending by date_range_start, undated passages placed after dated
    - Add `## Conversation with {name} (Your private conversation)` header per section
    - Separate sections with `---` on its own line
    - Add `[YYYY-MM-DD to YYYY-MM-DD]` date labels for passages with both start and end dates
    - Include all passages without discarding any
    - Implement `_format_date_label(chunk: SearchResult) -> str | None`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [ ] 7.2 Write property test for Participant Grouping Completeness
    - **Property 1: Participant Grouping Completeness**
    - **Validates: Requirements 1.1, 1.6**
    - Generate SearchResult lists with varying participant_name metadata
    - Assert every input passage appears exactly once in formatted output, grouped under correct header
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.3 Write property test for Participant Section Header Format
    - **Property 2: Participant Section Header Format**
    - **Validates: Requirements 1.2**
    - Generate participant name strings; assert section headers match `## Conversation with {name} (Your private conversation)`
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.4 Write property test for Section Separation
    - **Property 3: Section Separation**
    - **Validates: Requirements 1.3**
    - Generate SearchResults from 2+ participants; assert `---` between adjacent sections but not before first or after last
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.5 Write property test for Chronological Ordering Within Participant Sections
    - **Property 4: Chronological Ordering Within Participant Sections**
    - **Validates: Requirements 1.4, 1.7**
    - Generate same-participant results with varying dates; assert dated passages are in ascending order, undated after dated
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.6 Write property test for Participant Section Ordering by Score
    - **Property 5: Participant Section Ordering by Score**
    - **Validates: Requirements 1.8**
    - Generate multi-participant results with varying scores; assert sections appear in descending max-score order
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.7 Write property test for Date Range Annotation Format
    - **Property 6: Date Range Annotation Format**
    - **Validates: Requirements 1.5**
    - Generate SearchResults with and without date metadata; assert format `[YYYY-MM-DD to YYYY-MM-DD]` or no annotation
    - Place test in `tests/test_property_orchestrator.py`

  - [ ] 7.8 Write property test for Backward-Compatible Chunk Presentation
    - **Property 16: Backward-Compatible Chunk Presentation**
    - **Validates: Requirements 6.1**
    - Generate SearchResults whose text lacks speaker prefix patterns; assert text passes through verbatim
    - Place test in `tests/test_property_orchestrator.py`

- [ ] 8. Update system prompt and conditional prefix instruction
  - [ ] 8.1 Update `_SYSTEM_TEMPLATE` and add conditional prefix explanation
    - Replace existing `_SYSTEM_TEMPLATE` with the new prompt that explains `## Conversation with ...` sections as separate private conversations
    - Include statement that participants did not converse with each other
    - Include explanation of `[You]:` and `[{Name}]:` prefixes
    - Ensure the system prompt is always included regardless of whether current context has prefixes (handles mixed data)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 8.2 Write property test for Conditional Speaker Prefix Prompt Instruction
    - **Property 18: Conditional Speaker Prefix Prompt Instruction**
    - **Validates: Requirements 7.3**
    - Generate formatted context containing at least one speaker prefix pattern
    - Assert the system prompt includes explanation of prefix meaning
    - Place test in `tests/test_property_orchestrator.py`

- [ ] 9. Ensure backward compatibility with mixed old/new chunks
  - [ ] 9.1 Verify orchestrator handles mixed-format context without errors
    - Ensure `_format_context` works when some SearchResults have prefixed text and others have plain text
    - Ensure `query()` returns valid `GenerationResult` with non-empty `text` and non-empty `source_chunks` regardless of prefix presence
    - Old chunks (without prefixes) pass through verbatim in their participant section
    - _Requirements: 6.1, 6.2, 6.5_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 18 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Hypothesis for property-based testing (already installed and configured)
- Existing test files: `tests/test_property_pdf_parser.py`, `tests/test_property_chunker.py`, `tests/test_property_orchestrator.py` already exist and should be extended
- Custom Hypothesis generators (`search_result_generator`, `message_with_role_generator`, `text_block_generator`, `parsed_conversation_generator`) should be placed in a shared `tests/conftest.py` or within each test file

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.5", "3.2"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5", "5.1"] },
    { "id": 5, "tasks": ["5.2"] },
    { "id": 6, "tasks": ["5.3", "5.4", "5.5", "7.1"] },
    { "id": 7, "tasks": ["7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] }
  ]
}
```
