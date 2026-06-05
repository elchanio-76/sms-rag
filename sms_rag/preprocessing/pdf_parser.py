"""PDF parser for extracting messages from conversation PDFs using PyMuPDF."""

import logging
import re
from datetime import datetime
from pathlib import Path

import pymupdf

from sms_rag.shared.models import Message, ParsedConversation, TextBlock

logger = logging.getLogger(__name__)

# Greek day abbreviations and full names
_GREEK_DAYS_ABBR = r"(?:Δευ|Τρί|Τετ|Πέµ|Παρ|Σάβ|Κυρ)"
_GREEK_DAYS_FULL = r"(?:Δευτέρα|Τρίτη|Τετάρτη|Πέµπτη|Παρασκευή|Σάββατο|Κυριακή)"
_GREEK_MONTHS = r"(?:Ιαν|Φεβ|Μαρ|Απρ|Μαΐ|Ιουν|Ιουλ|Αυγ|Σεπ|Οκτ|Νοε|Δεκ)"

# Date/time patterns found in the PDFs:
# "Παρ 20 Φεβ, 10:30 πµ" - abbreviated day + day number + month + time
# "Παρασκευή 6:13 µµ" - full day name + time (no date)
# "Δευ 15 Δεκ, 7:31 µµ" - abbreviated day + day number + month + time
_DATE_PATTERN_FULL = re.compile(
    rf"^({_GREEK_DAYS_ABBR})\s+(\d{{1,2}})\s+({_GREEK_MONTHS}),?\s+(\d{{1,2}}):(\d{{2}})\s*(πµ|µµ)$",
    re.MULTILINE,
)

_DATE_PATTERN_DAY_ONLY = re.compile(
    rf"^({_GREEK_DAYS_FULL})\s+(\d{{1,2}}):(\d{{2}})\s*(πµ|µµ)$",
    re.MULTILINE,
)

# Message type indicators
_MESSAGE_TYPE_PATTERN = re.compile(
    r"^(iMessage|Γραπτό µήνυµα\s*•\s*SMS|Γραπτό µήνυµα\s*•\s*RCS)$",
    re.MULTILINE,
)

# Phone number pattern (international format)
_PHONE_PATTERN = re.compile(r"\+\d{10,15}")

# Lines to skip (system indicators, not message content)
_SKIP_PATTERNS = [
    re.compile(r"^EIXATE\s+\d+\s+KΛHƩH:"),
    re.compile(r"^Αναγνώστηκε\s*\d*/?.*$"),
    re.compile(r"^\(\d+\)\s+\d{2}/\d{2}\s+\d{2}:\d{2}$"),
]

# Delivery receipt pattern - acts as a message boundary separator
_DELIVERY_RECEIPT_PATTERN = re.compile(r"^Παραδόθηκε$")

# Greek month name to month number mapping
_MONTH_MAP = {
    "Ιαν": 1,
    "Φεβ": 2,
    "Μαρ": 3,
    "Απρ": 4,
    "Μαΐ": 5,
    "Ιουν": 6,
    "Ιουλ": 7,
    "Αυγ": 8,
    "Σεπ": 9,
    "Οκτ": 10,
    "Νοε": 11,
    "Δεκ": 12,
}


def _parse_timestamp_full(match: re.Match) -> datetime | None:
    """Parse a full date/time match (day abbr + day num + month + time)."""
    try:
        _day_name, day_num, month_name, hour, minute, period = match.groups()
        month = _MONTH_MAP.get(month_name)
        if month is None:
            return None
        hour_int = int(hour)
        minute_int = int(minute)
        # Convert to 24-hour format
        if period == "µµ" and hour_int != 12:
            hour_int += 12
        elif period == "πµ" and hour_int == 12:
            hour_int = 0
        # We don't have year info - use a default year
        # The actual year isn't in the PDF, so we use a reasonable default
        return datetime(2024, month, int(day_num), hour_int, minute_int)
    except (ValueError, TypeError):
        return None


def _parse_timestamp_day_only(match: re.Match) -> datetime | None:
    """Parse a day-only date/time match (full day name + time, no date)."""
    try:
        _day_name, hour, minute, period = match.groups()
        hour_int = int(hour)
        minute_int = int(minute)
        if period == "µµ" and hour_int != 12:
            hour_int += 12
        elif period == "πµ" and hour_int == 12:
            hour_int = 0
        # No date available, return None for timestamp
        return None
    except (ValueError, TypeError):
        return None


def _normalize_message_type(raw_type: str) -> str:
    """Normalize message type string to standard form."""
    raw_type = raw_type.strip()
    if raw_type == "iMessage":
        return "iMessage"
    elif "SMS" in raw_type:
        return "SMS"
    elif "RCS" in raw_type:
        return "RCS"
    return raw_type


def _is_skip_line(line: str) -> bool:
    """Check if a line should be skipped (system indicator, not message content)."""
    for pattern in _SKIP_PATTERNS:
        if pattern.match(line):
            return True
    return False


class PDFParser:
    """Parses conversation PDFs into structured Message objects."""

    def __init__(
        self, x_threshold_ratio: float = 0.5, ambiguity_margin: float = 10.0
    ) -> None:
        """Initialize PDFParser with coordinate classification parameters.

        Args:
            x_threshold_ratio: Ratio of page width to use as left/right threshold.
                Default 0.5 (50% of page width).
            ambiguity_margin: Points within which a block is considered ambiguous.
                Default 10.0 points.
        """
        self._x_threshold_ratio = x_threshold_ratio
        self._ambiguity_margin = ambiguity_margin
        self._receipt_followed_indices: set[int] = set()

    def _extract_text_blocks_with_coords(self, page: pymupdf.Page) -> list[TextBlock]:
        """Extract text blocks with x-coordinate positions from a page.

        Uses page.get_text("dict") to retrieve block-level spatial information.
        Each block's left edge (x0) is captured as the x_position.

        Args:
            page: A PyMuPDF Page object.

        Returns:
            List of TextBlock objects with text content, x0 coordinate, and page number.
        """
        page_dict = page.get_text("dict")
        page_number = page.number
        text_blocks: list[TextBlock] = []

        for block in page_dict.get("blocks", []):
            # Only process text blocks (type 0), skip image blocks (type 1)
            if block.get("type") != 0:
                continue

            # Extract text from all lines and spans within the block
            block_text_parts: list[str] = []
            for line in block.get("lines", []):
                line_text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                if line_text.strip():
                    block_text_parts.append(line_text.strip())

            block_text = "\n".join(block_text_parts)
            if not block_text.strip():
                continue

            # x0 is the left edge of the block's bounding box
            x0 = block["bbox"][0]

            text_blocks.append(
                TextBlock(
                    text=block_text,
                    x_position=x0,
                    page_number=page_number,
                )
            )

        return text_blocks

    def _classify_speaker_by_coordinates(
        self, text_blocks: list[TextBlock], page_width: float
    ) -> dict[str, str] | None:
        """Classify text blocks as left/right aligned based on x-coordinate.

        Returns a mapping of text -> speaker_role, or None if classification
        is unreliable (<80% of non-ambiguous blocks classifiable).

        Args:
            text_blocks: List of TextBlock objects with x-coordinate positions.
            page_width: Width of the PDF page in points.

        Returns:
            Dict mapping block text to "sent" or "received", or None if
            classification is unreliable.
        """
        if not text_blocks:
            return None

        threshold = page_width * self._x_threshold_ratio

        classified: dict[str, str] = {}
        total_blocks = len(text_blocks)

        for block in text_blocks:
            distance_from_threshold = abs(block.x_position - threshold)

            # Blocks within ambiguity margin are excluded from classification
            if distance_from_threshold <= self._ambiguity_margin:
                continue

            if block.x_position < threshold:
                classified[block.text] = "received"
            else:  # x_position >= threshold (and outside ambiguity margin)
                classified[block.text] = "sent"

        # Check reliability: need at least 80% of non-ambiguous blocks
        # relative to total blocks to consider classification reliable
        non_ambiguous_count = len(classified)
        if non_ambiguous_count == 0:
            logger.warning(
                "Coordinate classification: all blocks are ambiguous "
                "(within %.1f points of threshold). Falling back to receipt heuristic.",
                self._ambiguity_margin,
            )
            return None

        classifiable_ratio = non_ambiguous_count / total_blocks
        if classifiable_ratio < 0.8:
            logger.warning(
                "Coordinate classification unreliable: only %.1f%% of "
                "blocks classifiable (threshold: 80%%). Falling back to receipt heuristic.",
                classifiable_ratio * 100,
            )
            return None

        return classified

    def parse(self, pdf_path: Path) -> ParsedConversation:
        """Parse a single PDF file into structured messages.

        Args:
            pdf_path: Path to the PDF file to parse.

        Returns:
            ParsedConversation with extracted messages and metadata.
            On error, returns an empty conversation with errors listed.
        """
        source_filename = pdf_path.name
        participant_name = pdf_path.stem

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            error_msg = f"Failed to open PDF '{source_filename}': {e}"
            logger.error(error_msg)
            return ParsedConversation(
                participant_name=participant_name,
                source_filename=source_filename,
                messages=[],
                errors=[error_msg],
            )

        try:
            # Check for encrypted/password-protected PDFs
            if doc.is_encrypted:
                error_msg = f"PDF '{source_filename}' is password-protected"
                logger.error(error_msg)
                doc.close()
                return ParsedConversation(
                    participant_name=participant_name,
                    source_filename=source_filename,
                    messages=[],
                    errors=[error_msg],
                )

            # Extract all text from all pages
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            doc.close()

            if not full_text.strip():
                error_msg = f"PDF '{source_filename}' contains no extractable text"
                logger.warning(error_msg)
                return ParsedConversation(
                    participant_name=participant_name,
                    source_filename=source_filename,
                    messages=[],
                    errors=[error_msg],
                )

            messages = self._extract_messages(full_text)

            return ParsedConversation(
                participant_name=participant_name,
                source_filename=source_filename,
                messages=messages,
                errors=[],
            )

        except Exception as e:
            error_msg = f"Error processing PDF '{source_filename}': {e}"
            logger.error(error_msg)
            if not doc.is_closed:
                doc.close()
            return ParsedConversation(
                participant_name=participant_name,
                source_filename=source_filename,
                messages=[],
                errors=[error_msg],
            )

    def _extract_messages(self, text: str) -> list[Message]:
        """Extract messages from raw PDF text.

        Detects message boundaries using date/time stamp patterns and
        message type indicators. Groups text between boundaries into
        individual messages.

        Also populates self._receipt_followed_indices with the set of
        message indices that were immediately followed by a delivery receipt
        in document order (used by _assign_speaker_roles_by_receipt).
        """
        lines = text.split("\n")
        messages: list[Message] = []
        self._receipt_followed_indices: set[int] = set()

        current_message_type: str | None = None
        current_timestamp: datetime | None = None
        current_phone: str | None = None
        current_text_lines: list[str] = []
        in_message = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for message type indicator (appears before date)
            type_match = _MESSAGE_TYPE_PATTERN.match(line)
            if type_match:
                # Save any accumulated message before starting new context
                if in_message and current_text_lines:
                    messages.append(
                        Message(
                            text="\n".join(current_text_lines).strip(),
                            timestamp=current_timestamp,
                            message_type=current_message_type,
                            phone_number=current_phone,
                        )
                    )
                    current_text_lines = []

                current_message_type = _normalize_message_type(type_match.group(1))
                current_timestamp = None
                current_phone = None
                in_message = False
                i += 1
                continue

            # Check for full date pattern (day abbr + number + month + time)
            date_full_match = _DATE_PATTERN_FULL.match(line)
            if date_full_match:
                # Save any accumulated message
                if in_message and current_text_lines:
                    messages.append(
                        Message(
                            text="\n".join(current_text_lines).strip(),
                            timestamp=current_timestamp,
                            message_type=current_message_type,
                            phone_number=current_phone,
                        )
                    )
                    current_text_lines = []

                current_timestamp = _parse_timestamp_full(date_full_match)
                # Keep current_phone from context (e.g., phone seen after type indicator)
                in_message = True
                i += 1
                continue

            # Check for day-only date pattern (full day name + time)
            date_day_match = _DATE_PATTERN_DAY_ONLY.match(line)
            if date_day_match:
                # Save any accumulated message
                if in_message and current_text_lines:
                    messages.append(
                        Message(
                            text="\n".join(current_text_lines).strip(),
                            timestamp=current_timestamp,
                            message_type=current_message_type,
                            phone_number=current_phone,
                        )
                    )
                    current_text_lines = []

                current_timestamp = _parse_timestamp_day_only(date_day_match)
                # Keep current_phone from context
                in_message = True
                i += 1
                continue

            # Check for phone numbers in the line
            phone_match = _PHONE_PATTERN.search(line)
            if phone_match:
                current_phone = phone_match.group(0)
                # If this line is ONLY a phone number (standalone), skip it as content
                if line.strip() == phone_match.group(0):
                    i += 1
                    continue
                # If the phone is embedded in text, continue to add the line as content

            # Delivery receipt acts as a boundary - save current message and mark it
            if _DELIVERY_RECEIPT_PATTERN.match(line):
                if in_message and current_text_lines:
                    messages.append(
                        Message(
                            text="\n".join(current_text_lines).strip(),
                            timestamp=current_timestamp,
                            message_type=current_message_type,
                            phone_number=current_phone,
                        )
                    )
                    # Track that this message was followed by a delivery receipt
                    self._receipt_followed_indices.add(len(messages) - 1)
                    current_text_lines = []
                    # Keep current context (type, timestamp) for the response
                    # but mark that we're ready for new text
                    in_message = True
                # If no message was accumulated (orphan receipt), ignore it (Req 3.5)
                i += 1
                continue

            # Skip system indicator lines
            if _is_skip_line(line):
                i += 1
                continue

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Accumulate message text
            if in_message:
                current_text_lines.append(line)
            else:
                # Text before any date marker - still capture it
                # This handles cases where a message type appears but no date follows immediately
                if current_message_type is not None:
                    in_message = True
                    current_text_lines.append(line)

            i += 1

        # Don't forget the last message
        if current_text_lines:
            messages.append(
                Message(
                    text="\n".join(current_text_lines).strip(),
                    timestamp=current_timestamp,
                    message_type=current_message_type,
                    phone_number=current_phone,
                )
            )

        # Filter out empty messages
        messages = [m for m in messages if m.text.strip()]

        return messages

    def _assign_speaker_roles_by_receipt(
        self, messages: list[Message]
    ) -> list[Message]:
        """Fallback: assign speaker roles using delivery receipt heuristic.

        Uses the receipt tracking information populated by _extract_messages
        (self._receipt_followed_indices) to determine which messages were sent.

        Rules:
        - Messages immediately followed by a delivery receipt ("Παραδόθηκε")
          in document order are tagged as "sent" (Req 3.1).
        - Messages not followed by a receipt and without coordinate-based
          classification get "unknown" (Req 3.2).
        - A "sent" assignment is never made unless justified by a receipt
          or coordinate classification (Req 3.3).
        - Orphan receipts (no preceding message) are already ignored during
          extraction (Req 3.5).

        Args:
            messages: List of Message objects in document order, as produced
                by _extract_messages.

        Returns:
            The same list of messages with speaker_role assigned.
        """
        receipt_indices = getattr(self, "_receipt_followed_indices", set())

        for idx, message in enumerate(messages):
            if idx in receipt_indices:
                message.speaker_role = "sent"
            else:
                # Only assign "unknown" if no coordinate-based role was already set.
                # If coordinate classification previously set a role, preserve it.
                # (In pure receipt-heuristic mode, all non-receipt messages get "unknown".)
                if message.speaker_role == "unknown":
                    message.speaker_role = "unknown"

        return messages
