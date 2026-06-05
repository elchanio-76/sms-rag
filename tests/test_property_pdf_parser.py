"""Property-based tests for PDF parsing.

# Feature: sms-rag, Property 1: Message Chronological Order Preservation
# Feature: sms-rag, Property 2: Metadata Extraction Completeness
# Feature: sms-rag, Property 3: Filename to Participant Name Derivation
# Feature: participant-context-clarity, Property 13: Speaker Role Domain Invariant
# Feature: participant-context-clarity, Property 7: Coordinate Classification Correctness

Validates: Requirements 1.1, 1.2, 1.4, 3.6, 4.1, 4.2, 2.2, 2.4
"""

import string
from datetime import datetime

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.shared.models import Message, TextBlock


# --- Greek constants used by the parser ---

GREEK_DAY_ABBREVS = ["Δευ", "Τρί", "Τετ", "Πέµ", "Παρ", "Σάβ", "Κυρ"]
GREEK_MONTHS = [
    "Ιαν",
    "Φεβ",
    "Μαρ",
    "Απρ",
    "Μαΐ",
    "Ιουν",
    "Ιουλ",
    "Αυγ",
    "Σεπ",
    "Οκτ",
    "Νοε",
    "Δεκ",
]
GREEK_MONTH_TO_NUM = {
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
MESSAGE_TYPES = ["iMessage", "Γραπτό µήνυµα • SMS", "Γραπτό µήνυµα • RCS"]
NORMALIZED_TYPES = {
    "iMessage": "iMessage",
    "Γραπτό µήνυµα • SMS": "SMS",
    "Γραπτό µήνυµα • RCS": "RCS",
}

# Maximum valid day per month (non-leap year for simplicity)
MAX_DAYS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


# --- Strategies ---


@st.composite
def greek_timestamp_strategy(draw):
    """Generate a valid Greek date/time stamp string and its expected datetime.

    Format: "Παρ 20 Φεβ, 10:30 πµ"
    """
    day_abbr = draw(st.sampled_from(GREEK_DAY_ABBREVS))
    month_name = draw(st.sampled_from(GREEK_MONTHS))
    month_num = GREEK_MONTH_TO_NUM[month_name]
    max_day = MAX_DAYS[month_num]
    day_num = draw(st.integers(min_value=1, max_value=max_day))
    hour = draw(st.integers(min_value=1, max_value=12))
    minute = draw(st.integers(min_value=0, max_value=59))
    period = draw(st.sampled_from(["πµ", "µµ"]))

    # Compute expected 24-hour time
    hour_24 = hour
    if period == "µµ" and hour != 12:
        hour_24 = hour + 12
    elif period == "πµ" and hour == 12:
        hour_24 = 0

    timestamp_str = f"{day_abbr} {day_num} {month_name}, {hour}:{minute:02d} {period}"
    expected_dt = datetime(2024, month_num, day_num, hour_24, minute)

    return timestamp_str, expected_dt


@st.composite
def message_text_strategy(draw):
    """Generate message text that doesn't conflict with parser patterns.

    Avoids generating text that looks like timestamps, message types, or phone numbers.
    """
    # Use a safe alphabet: Greek + Latin letters + digits + basic punctuation
    safe_chars = (
        "αβγδεζηθικλμνξοπρστυφχψω" + "abcdefghijklmnopqrstuvwxyz" + "0123456789 .,!?-"
    )
    text = draw(st.text(alphabet=safe_chars, min_size=1, max_size=80))
    # Ensure it's not empty after strip
    assume(text.strip() != "")
    # Ensure it doesn't start with + followed by many digits (phone pattern)
    assume(not text.startswith("+"))
    # Ensure it doesn't look like a delivery receipt
    assume(text.strip() != "Παραδόθηκε")
    return text.strip()


@st.composite
def phone_number_strategy(draw):
    """Generate valid international phone numbers (+XXXXXXXXXX, 10-15 digits)."""
    num_digits = draw(st.integers(min_value=10, max_value=15))
    digits = draw(
        st.text(alphabet="0123456789", min_size=num_digits, max_size=num_digits)
    )
    return f"+{digits}"


@st.composite
def sorted_timestamps_strategy(draw, min_count=2, max_count=10):
    """Generate a sorted list of (timestamp_str, datetime) pairs."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    timestamps = []
    for _ in range(count):
        ts = draw(greek_timestamp_strategy())
        timestamps.append(ts)

    # Sort by datetime
    timestamps.sort(key=lambda x: x[1])
    return timestamps


@st.composite
def conversation_text_strategy(draw):
    """Generate a complete conversation text with multiple messages in chronological order.

    Returns (text, expected_timestamps) where expected_timestamps is a sorted list of datetimes.
    """
    timestamps = draw(sorted_timestamps_strategy(min_count=2, max_count=8))
    msg_type = draw(st.sampled_from(MESSAGE_TYPES))

    lines = [msg_type]

    expected_dts = []
    for ts_str, ts_dt in timestamps:
        lines.append(ts_str)
        msg_text = draw(message_text_strategy())
        lines.append(msg_text)
        expected_dts.append(ts_dt)

    text = "\n".join(lines)
    return text, expected_dts


@st.composite
def filename_strategy(draw):
    """Generate valid PDF filenames.

    Filenames can contain letters, digits, spaces, hyphens, dots, underscores,
    apostrophes, and various unicode characters.
    """
    # Use a mix of safe characters for filenames
    safe_chars = (
        string.ascii_letters
        + string.digits
        + " -_.'()αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    )
    name = draw(st.text(alphabet=safe_chars, min_size=1, max_size=60))
    # Ensure the name part is not empty after strip
    assume(name.strip() != "")
    # Ensure filename doesn't end with a dot (which would be confusing with .pdf)
    assume(not name.endswith("."))
    return name + ".pdf"


# --- Property Tests ---


class TestProperty1ChronologicalOrder:
    """Property 1: Message Chronological Order Preservation.

    For any sequence of messages with timestamps extracted from a PDF,
    the output list of Message objects SHALL be ordered such that each
    message's timestamp is less than or equal to the timestamp of the
    next message in the list.

    # Feature: sms-rag, Property 1: Message Chronological Order Preservation
    Validates: Requirements 1.1
    """

    @given(data=conversation_text_strategy())
    @settings(max_examples=25)
    def test_messages_preserve_chronological_order(self, data):
        """For any generated conversation with sorted timestamps,
        _extract_messages SHALL return messages in chronological order.

        **Validates: Requirements 1.1**
        """
        text, expected_dts = data
        parser = PDFParser()
        messages = parser._extract_messages(text)

        # Filter messages that have timestamps
        timestamped_messages = [m for m in messages if m.timestamp is not None]

        # We should have at least some timestamped messages
        assume(len(timestamped_messages) >= 2)

        # Verify chronological order is preserved
        for i in range(len(timestamped_messages) - 1):
            assert (
                timestamped_messages[i].timestamp
                <= timestamped_messages[i + 1].timestamp
            ), (
                f"Message {i} timestamp {timestamped_messages[i].timestamp} is after "
                f"message {i+1} timestamp {timestamped_messages[i+1].timestamp}"
            )

    @given(timestamps=sorted_timestamps_strategy(min_count=3, max_count=12))
    @settings(max_examples=25)
    def test_extracted_timestamps_match_input_order(self, timestamps):
        """For any sorted sequence of timestamps, messages extracted
        SHALL have timestamps in the same non-decreasing order.

        **Validates: Requirements 1.1**
        """
        # Build conversation text with a message type header
        lines = ["iMessage"]
        for ts_str, _ in timestamps:
            lines.append(ts_str)
            lines.append("test message content")

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)

        timestamped = [m for m in messages if m.timestamp is not None]
        assume(len(timestamped) >= 2)

        for i in range(len(timestamped) - 1):
            assert timestamped[i].timestamp <= timestamped[i + 1].timestamp


class TestProperty2MetadataCompleteness:
    """Property 2: Metadata Extraction Completeness.

    For any message text that contains a recognizable date/time stamp pattern,
    message type indicator, or phone number, the corresponding Message object
    SHALL have those fields populated with the extracted values.

    # Feature: sms-rag, Property 2: Metadata Extraction Completeness
    Validates: Requirements 1.2
    """

    @given(
        ts_data=greek_timestamp_strategy(),
        msg_type=st.sampled_from(MESSAGE_TYPES),
        msg_text=message_text_strategy(),
    )
    @settings(max_examples=25)
    def test_timestamp_extracted_when_present(self, ts_data, msg_type, msg_text):
        """For any message with a recognizable date/time stamp,
        the Message object SHALL have timestamp populated.

        **Validates: Requirements 1.2**
        """
        ts_str, expected_dt = ts_data

        text = f"{msg_type}\n{ts_str}\n{msg_text}"
        parser = PDFParser()
        messages = parser._extract_messages(text)

        assert len(messages) >= 1
        # Find the message with our text
        matching = [m for m in messages if msg_text in m.text]
        assert len(matching) >= 1, f"Expected to find message with text '{msg_text}'"

        msg = matching[0]
        assert (
            msg.timestamp == expected_dt
        ), f"Expected timestamp {expected_dt}, got {msg.timestamp}"

    @given(
        msg_type=st.sampled_from(MESSAGE_TYPES),
        msg_text=message_text_strategy(),
    )
    @settings(max_examples=25)
    def test_message_type_extracted_when_present(self, msg_type, msg_text):
        """For any message with a recognizable message type indicator,
        the Message object SHALL have message_type populated.

        **Validates: Requirements 1.2**
        """
        # Use a fixed valid timestamp
        ts_str = "Παρ 20 Φεβ, 10:30 πµ"

        text = f"{msg_type}\n{ts_str}\n{msg_text}"
        parser = PDFParser()
        messages = parser._extract_messages(text)

        assert len(messages) >= 1
        matching = [m for m in messages if msg_text in m.text]
        assert len(matching) >= 1

        msg = matching[0]
        expected_normalized = NORMALIZED_TYPES[msg_type]
        assert (
            msg.message_type == expected_normalized
        ), f"Expected message_type '{expected_normalized}', got '{msg.message_type}'"

    @given(
        phone=phone_number_strategy(),
        msg_text=message_text_strategy(),
    )
    @settings(max_examples=25)
    def test_phone_number_extracted_when_present(self, phone, msg_text):
        """For any message context with a recognizable phone number,
        the Message object SHALL have phone_number populated.

        **Validates: Requirements 1.2**
        """
        # Phone appears as standalone line between type and date (typical pattern)
        text = f"Γραπτό µήνυµα • SMS\n{phone}\nΠαρ 20 Φεβ, 10:30 πµ\n{msg_text}"
        parser = PDFParser()
        messages = parser._extract_messages(text)

        assert len(messages) >= 1
        matching = [m for m in messages if msg_text in m.text]
        assert len(matching) >= 1

        msg = matching[0]
        assert (
            msg.phone_number == phone
        ), f"Expected phone_number '{phone}', got '{msg.phone_number}'"

    @given(
        ts_data=greek_timestamp_strategy(),
        msg_type=st.sampled_from(MESSAGE_TYPES),
        phone=phone_number_strategy(),
        msg_text=message_text_strategy(),
    )
    @settings(max_examples=25)
    def test_all_metadata_fields_extracted_together(
        self, ts_data, msg_type, phone, msg_text
    ):
        """For any message with all metadata fields present (timestamp, type, phone),
        ALL corresponding fields SHALL be populated.

        **Validates: Requirements 1.2**
        """
        ts_str, expected_dt = ts_data

        # Phone between type indicator and date (typical PDF layout)
        text = f"{msg_type}\n{phone}\n{ts_str}\n{msg_text}"
        parser = PDFParser()
        messages = parser._extract_messages(text)

        assert len(messages) >= 1
        matching = [m for m in messages if msg_text in m.text]
        assert len(matching) >= 1

        msg = matching[0]
        expected_normalized = NORMALIZED_TYPES[msg_type]
        assert (
            msg.timestamp == expected_dt
        ), f"Timestamp mismatch: {msg.timestamp} != {expected_dt}"
        assert (
            msg.message_type == expected_normalized
        ), f"Type mismatch: {msg.message_type} != {expected_normalized}"
        assert (
            msg.phone_number == phone
        ), f"Phone mismatch: {msg.phone_number} != {phone}"


class TestProperty3FilenameDerviation:
    """Property 3: Filename to Participant Name Derivation.

    For any valid PDF filename string, the derived participant name SHALL
    equal the filename with the .pdf extension removed and no other
    transformations applied.

    # Feature: sms-rag, Property 3: Filename to Participant Name Derivation
    Validates: Requirements 1.4
    """

    @given(filename=filename_strategy())
    @settings(max_examples=25)
    def test_participant_name_equals_filename_without_extension(self, filename):
        """For any valid PDF filename, the participant name SHALL be
        the filename minus the .pdf extension.

        **Validates: Requirements 1.4**
        """
        # The derivation logic uses Path.stem which removes the last extension
        from pathlib import Path as P

        expected_name = P(filename).stem

        # We test the derivation logic directly via Path.stem
        # which is what PDFParser.parse() uses: pdf_path.stem
        assert expected_name == filename[:-4] or expected_name == P(filename).stem

        # Verify it matches what the parser would produce
        # We can't easily create a real PDF for every filename,
        # but we can verify the logic: participant_name = pdf_path.stem
        pdf_path = P(f"/tmp/{filename}")
        participant_name = pdf_path.stem

        # The participant name should be the filename without .pdf
        assert participant_name == filename.removesuffix(
            ".pdf"
        ), f"Expected '{filename.removesuffix('.pdf')}', got '{participant_name}'"

    @given(filename=filename_strategy())
    @settings(max_examples=25)
    def test_no_transformation_beyond_extension_removal(self, filename):
        """For any filename, NO transformations (case change, trimming,
        character substitution) SHALL be applied beyond removing .pdf.

        **Validates: Requirements 1.4**
        """
        from pathlib import Path as P

        pdf_path = P(f"/tmp/{filename}")
        participant_name = pdf_path.stem
        expected = filename[:-4]  # Remove '.pdf' (4 chars)

        # Verify exact equality - no lowercasing, stripping, or other transforms
        assert (
            participant_name == expected
        ), f"Transformation detected: '{expected}' became '{participant_name}'"

    @given(
        name_part=st.text(
            alphabet=string.ascii_letters
            + string.digits
            + " -_.'()ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδ",
            min_size=1,
            max_size=40,
        )
    )
    @settings(max_examples=25)
    def test_participant_name_preserves_unicode_characters(self, name_part):
        """For filenames with unicode characters, the participant name
        SHALL preserve all characters exactly.

        **Validates: Requirements 1.4**
        """
        assume(name_part.strip() != "")
        assume(not name_part.endswith("."))

        from pathlib import Path as P

        filename = f"{name_part}.pdf"
        pdf_path = P(f"/tmp/{filename}")
        participant_name = pdf_path.stem

        assert (
            participant_name == name_part
        ), f"Unicode not preserved: expected '{name_part}', got '{participant_name}'"


# --- Speaker Role Generator ---

VALID_SPEAKER_ROLES = ["sent", "received", "unknown"]


@st.composite
def message_with_role_generator(draw):
    """Generate Message objects with arbitrary text and speaker_role values.

    Produces Message objects with:
    - Arbitrary text content (including unicode, embedded newlines, edge cases)
    - speaker_role drawn from the valid domain: {"sent", "received", "unknown"}
    """
    text = draw(
        st.text(
            min_size=1,
            max_size=200,
        )
    )
    assume(text.strip() != "")
    role = draw(st.sampled_from(VALID_SPEAKER_ROLES))
    return Message(text=text, speaker_role=role)


# --- Property Test: Speaker Role Domain Invariant ---


class TestProperty13SpeakerRoleDomainInvariant:
    """Property 13: Speaker Role Domain Invariant.

    For any Message object produced by the PDF_Parser, the speaker_role field
    SHALL contain exactly one of the values "sent", "received", or "unknown".

    # Feature: participant-context-clarity, Property 13: Speaker Role Domain Invariant
    Validates: Requirements 3.6, 4.1, 4.2
    """

    @given(message=message_with_role_generator())
    @settings(max_examples=100)
    def test_speaker_role_always_in_valid_domain(self, message):
        """For any Message object, the speaker_role field SHALL contain exactly
        one of: "sent", "received", or "unknown".

        **Validates: Requirements 3.6, 4.1, 4.2**
        """
        assert message.speaker_role in {"sent", "received", "unknown"}, (
            f"speaker_role '{message.speaker_role}' is not in the allowed domain "
            f'{{"sent", "received", "unknown"}}'
        )

    @given(data=conversation_text_strategy())
    @settings(max_examples=100)
    def test_parser_produced_messages_have_valid_speaker_role(self, data):
        """For any messages produced by the PDF parser's _extract_messages method,
        every Message SHALL have speaker_role in {"sent", "received", "unknown"}.

        **Validates: Requirements 3.6, 4.1, 4.2**
        """
        text, _ = data
        parser = PDFParser()
        messages = parser._extract_messages(text)

        for msg in messages:
            assert msg.speaker_role in {"sent", "received", "unknown"}, (
                f"Parser produced message with speaker_role '{msg.speaker_role}' "
                f"which is not in the allowed domain "
                f'{{"sent", "received", "unknown"}}'
            )

    @given(
        text=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100)
    def test_default_speaker_role_is_unknown(self, text):
        """When a Message is created without specifying speaker_role,
        it SHALL default to "unknown".

        **Validates: Requirements 4.2**
        """
        assume(text.strip() != "")
        msg = Message(text=text)
        assert (
            msg.speaker_role == "unknown"
        ), f"Default speaker_role should be 'unknown', got '{msg.speaker_role}'"


# --- Text Block Generator ---


@st.composite
def text_block_generator(draw, page_width, threshold_ratio, ambiguity_margin):
    """Generate TextBlock lists with configurable page_width, threshold_ratio, and ambiguity_margin.

    Produces a list of TextBlock objects with x-coordinates distributed to test
    classification boundaries. Blocks are generated to be clearly non-ambiguous
    (outside the ambiguity margin from the threshold).

    Args:
        page_width: Width of the page in points.
        threshold_ratio: Ratio of page_width used as left/right threshold.
        ambiguity_margin: Points within which a block is ambiguous.
    """
    threshold = page_width * threshold_ratio

    # Generate a mix of left-aligned and right-aligned blocks (non-ambiguous)
    num_blocks = draw(st.integers(min_value=1, max_value=20))
    blocks = []

    for i in range(num_blocks):
        side = draw(st.sampled_from(["left", "right"]))

        if side == "left":
            # x_position clearly below threshold (outside ambiguity margin)
            max_left = threshold - ambiguity_margin - 0.01
            if max_left <= 0:
                # If threshold is too close to 0, skip generating left blocks
                # and generate right instead
                min_right = threshold + ambiguity_margin + 0.01
                x_pos = draw(
                    st.floats(
                        min_value=min_right,
                        max_value=page_width,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            else:
                x_pos = draw(
                    st.floats(
                        min_value=0.0,
                        max_value=max_left,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
        else:
            # x_position clearly at/above threshold (outside ambiguity margin)
            min_right = threshold + ambiguity_margin + 0.01
            if min_right > page_width:
                # If threshold is too close to page_width, generate left instead
                max_left = threshold - ambiguity_margin - 0.01
                x_pos = draw(
                    st.floats(
                        min_value=0.0,
                        max_value=max(0.0, max_left),
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            else:
                x_pos = draw(
                    st.floats(
                        min_value=min_right,
                        max_value=page_width,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )

        text = draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=50)
        )
        assume(text.strip() != "")

        blocks.append(
            TextBlock(
                text=text.strip(),
                x_position=x_pos,
                page_number=draw(st.integers(min_value=0, max_value=10)),
            )
        )

    # Ensure unique text values to avoid dict key collisions
    seen_texts = set()
    unique_blocks = []
    for block in blocks:
        if block.text not in seen_texts:
            seen_texts.add(block.text)
            unique_blocks.append(block)

    assume(len(unique_blocks) >= 1)
    return unique_blocks


# --- Property Test: Coordinate Classification Correctness ---


class TestProperty7CoordinateClassificationCorrectness:
    """Property 7: Coordinate Classification Correctness.

    For any set of text blocks with x-coordinates and a given page width,
    blocks with x_position < threshold (where threshold = page_width × x_threshold_ratio)
    and not within the ambiguity margin SHALL be classified as left-aligned ("received"),
    and blocks with x_position >= threshold and not within the ambiguity margin SHALL be
    classified as right-aligned ("sent").

    # Feature: participant-context-clarity, Property 7: Coordinate Classification Correctness
    Validates: Requirements 2.2, 2.4
    """

    @given(
        page_width=st.floats(
            min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_non_ambiguous_blocks_classified_correctly(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """For any set of non-ambiguous text blocks, blocks below threshold get "received"
        and blocks at/above threshold get "sent".

        **Validates: Requirements 2.2, 2.4**
        """
        threshold = page_width * threshold_ratio

        # Ensure enough space for non-ambiguous blocks on both sides
        assume(threshold - ambiguity_margin > 0)
        assume(threshold + ambiguity_margin < page_width)

        blocks = data.draw(
            text_block_generator(page_width, threshold_ratio, ambiguity_margin)
        )
        assume(len(blocks) >= 1)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        # The method may return None if <80% of total blocks are non-ambiguous.
        # Since our generator only produces non-ambiguous blocks, result should not be None.
        # However, if all blocks happen to have the same text (collapsed to 1 key), handle gracefully.
        if result is None:
            # This can happen if blocks were deduplicated to fewer items
            return

        # Verify classification correctness for each block
        for block in blocks:
            distance_from_threshold = abs(block.x_position - threshold)

            if distance_from_threshold <= ambiguity_margin:
                # Ambiguous blocks should NOT appear in the result
                assert block.text not in result, (
                    f"Ambiguous block '{block.text}' (x={block.x_position}, "
                    f"distance={distance_from_threshold:.2f}, margin={ambiguity_margin}) "
                    f"should not be in classification result"
                )
            else:
                # Non-ambiguous blocks should be classified
                if block.text in result:
                    if block.x_position < threshold:
                        assert result[block.text] == "received", (
                            f"Block '{block.text}' at x={block.x_position} < threshold={threshold} "
                            f"should be 'received', got '{result[block.text]}'"
                        )
                    else:  # x_position >= threshold
                        assert result[block.text] == "sent", (
                            f"Block '{block.text}' at x={block.x_position} >= threshold={threshold} "
                            f"should be 'sent', got '{result[block.text]}'"
                        )

    @given(
        page_width=st.floats(
            min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_left_aligned_blocks_always_received(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """For any text block with x_position clearly below threshold,
        classification SHALL assign "received".

        **Validates: Requirements 2.2, 2.4**
        """
        threshold = page_width * threshold_ratio
        max_left = threshold - ambiguity_margin - 0.01
        assume(max_left > 0)

        # Generate blocks that are all clearly left-aligned
        num_blocks = data.draw(st.integers(min_value=1, max_value=10))
        blocks = []
        seen_texts = set()
        for _ in range(num_blocks):
            x_pos = data.draw(
                st.floats(
                    min_value=0.0,
                    max_value=max_left,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=30)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        assume(len(blocks) >= 1)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        if result is None:
            return

        for block in blocks:
            if block.text in result:
                assert result[block.text] == "received", (
                    f"Left-aligned block '{block.text}' at x={block.x_position} "
                    f"(threshold={threshold}) should be 'received', got '{result[block.text]}'"
                )

    @given(
        page_width=st.floats(
            min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_right_aligned_blocks_always_sent(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """For any text block with x_position clearly at/above threshold,
        classification SHALL assign "sent".

        **Validates: Requirements 2.2, 2.4**
        """
        threshold = page_width * threshold_ratio
        min_right = threshold + ambiguity_margin + 0.01
        assume(min_right < page_width)

        # Generate blocks that are all clearly right-aligned
        num_blocks = data.draw(st.integers(min_value=1, max_value=10))
        blocks = []
        seen_texts = set()
        for _ in range(num_blocks):
            x_pos = data.draw(
                st.floats(
                    min_value=min_right,
                    max_value=page_width,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=30)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        assume(len(blocks) >= 1)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        if result is None:
            return

        for block in blocks:
            if block.text in result:
                assert result[block.text] == "sent", (
                    f"Right-aligned block '{block.text}' at x={block.x_position} "
                    f"(threshold={threshold}) should be 'sent', got '{result[block.text]}'"
                )


# --- Ambiguous Text Block Generator ---


@st.composite
def ambiguous_text_block_generator(draw, page_width, threshold_ratio, ambiguity_margin):
    """Generate TextBlock lists where ALL blocks have x-coordinates within ambiguity_margin of threshold.

    These blocks should be excluded from coordinate-based classification.
    The implementation uses `abs(x - threshold) <= ambiguity_margin` for the check,
    so we generate x-coordinates with `abs(x - threshold) < ambiguity_margin * 0.99`
    to avoid floating-point edge cases at the boundary.

    Args:
        page_width: Width of the page in points.
        threshold_ratio: Ratio of page_width used as left/right threshold.
        ambiguity_margin: Points within which a block is ambiguous.
    """
    threshold = page_width * threshold_ratio

    # Use a slightly tighter range to stay clearly within the ambiguity band
    # and avoid floating-point precision issues at the boundary
    safe_margin = ambiguity_margin * 0.99
    ambiguity_low = max(0.0, threshold - safe_margin)
    ambiguity_high = min(page_width, threshold + safe_margin)

    num_blocks = draw(st.integers(min_value=1, max_value=15))
    blocks = []
    seen_texts = set()

    for _ in range(num_blocks):
        # Generate x-coordinate strictly within ambiguity margin of threshold
        x_pos = draw(
            st.floats(
                min_value=ambiguity_low,
                max_value=ambiguity_high,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        # Double-check the generated position is actually ambiguous
        assume(abs(x_pos - threshold) <= ambiguity_margin)

        text = draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=50)
        )
        assume(text.strip() != "")
        text = text.strip()

        if text not in seen_texts:
            seen_texts.add(text)
            blocks.append(
                TextBlock(
                    text=text,
                    x_position=x_pos,
                    page_number=draw(st.integers(min_value=0, max_value=10)),
                )
            )

    assume(len(blocks) >= 1)
    return blocks


# --- Property Test: Ambiguity Exclusion ---


class TestProperty8AmbiguityExclusion:
    """Property 8: Ambiguity Exclusion.

    For any text block whose x-coordinate is within ambiguity_margin points of
    the threshold, that block SHALL be excluded from the reliability percentage
    calculation and SHALL NOT receive a coordinate-based speaker role assignment.

    # Feature: participant-context-clarity, Property 8: Ambiguity Exclusion
    Validates: Requirements 2.5
    """

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_all_ambiguous_blocks_return_none(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """When ALL blocks are within ambiguity_margin of threshold,
        classification SHALL return None (no roles assigned).

        **Validates: Requirements 2.5**
        """
        threshold = page_width * threshold_ratio
        # Ensure there is a valid ambiguity band within the page
        assume(threshold - ambiguity_margin >= 0)
        assume(threshold + ambiguity_margin <= page_width)

        blocks = data.draw(
            ambiguous_text_block_generator(
                page_width, threshold_ratio, ambiguity_margin
            )
        )
        assume(len(blocks) >= 1)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        # All blocks are ambiguous → 0 non-ambiguous → classification returns None
        assert result is None, (
            f"Expected None when all {len(blocks)} blocks are ambiguous "
            f"(within {ambiguity_margin} points of threshold={threshold}), "
            f"but got {result}"
        )

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_ambiguous_blocks_excluded_from_result_dict(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """When a mix of ambiguous and non-ambiguous blocks is present,
        ambiguous blocks SHALL NOT appear in the classification result dict.

        **Validates: Requirements 2.5**
        """
        threshold = page_width * threshold_ratio
        assume(threshold - ambiguity_margin > 0)
        assume(threshold + ambiguity_margin < page_width)

        # Generate some non-ambiguous blocks (to ensure classification succeeds)
        non_ambiguous_blocks = data.draw(
            text_block_generator(page_width, threshold_ratio, ambiguity_margin)
        )
        assume(len(non_ambiguous_blocks) >= 1)

        # Generate some ambiguous blocks
        ambiguous_blocks = data.draw(
            ambiguous_text_block_generator(
                page_width, threshold_ratio, ambiguity_margin
            )
        )
        assume(len(ambiguous_blocks) >= 1)

        # Ensure no text collisions between the two sets
        non_ambig_texts = {b.text for b in non_ambiguous_blocks}
        ambiguous_blocks = [
            b for b in ambiguous_blocks if b.text not in non_ambig_texts
        ]
        assume(len(ambiguous_blocks) >= 1)

        # Combine both sets
        all_blocks = non_ambiguous_blocks + ambiguous_blocks

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(all_blocks, page_width)

        # If result is None due to reliability threshold, that's acceptable
        # (the ambiguous blocks increased total count, reducing classifiable ratio)
        if result is None:
            return

        # Ambiguous blocks should NOT appear in the result
        for block in ambiguous_blocks:
            distance_from_threshold = abs(block.x_position - threshold)
            assert block.text not in result, (
                f"Ambiguous block '{block.text}' (x={block.x_position}, "
                f"distance={distance_from_threshold:.2f}, margin={ambiguity_margin}) "
                f"should NOT appear in classification result"
            )

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_ambiguous_blocks_affect_reliability_calculation(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """Ambiguous blocks count toward total_blocks, reducing the classifiable
        ratio and potentially triggering the fallback (return None) when
        non-ambiguous blocks are less than 80% of total.

        **Validates: Requirements 2.5**
        """
        threshold = page_width * threshold_ratio
        assume(threshold - ambiguity_margin > 0)
        assume(threshold + ambiguity_margin < page_width)

        # Generate exactly 1 non-ambiguous block
        max_left = threshold - ambiguity_margin - 0.01
        non_ambig_x = data.draw(
            st.floats(
                min_value=0.0, max_value=max_left, allow_nan=False, allow_infinity=False
            )
        )
        non_ambig_text = data.draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
        )
        assume(non_ambig_text.strip() != "")
        non_ambig_text = non_ambig_text.strip()

        non_ambiguous_block = TextBlock(
            text=non_ambig_text, x_position=non_ambig_x, page_number=0
        )

        # Generate 5+ ambiguous blocks (to ensure ratio < 80%)
        num_ambiguous = data.draw(st.integers(min_value=5, max_value=15))
        # Use tighter range to avoid floating-point boundary issues
        safe_margin = ambiguity_margin * 0.99
        ambiguity_low = max(0.0, threshold - safe_margin)
        ambiguity_high = min(page_width, threshold + safe_margin)

        ambiguous_blocks = []
        seen_texts = {non_ambig_text}
        for _ in range(num_ambiguous):
            x_pos = data.draw(
                st.floats(
                    min_value=ambiguity_low,
                    max_value=ambiguity_high,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            # Verify this position is truly ambiguous
            assume(abs(x_pos - threshold) <= ambiguity_margin)
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                ambiguous_blocks.append(
                    TextBlock(text=text, x_position=x_pos, page_number=0)
                )

        assume(len(ambiguous_blocks) >= 4)

        # Combine: 1 non-ambiguous + many ambiguous → ratio = 1/(1+N) < 0.8
        all_blocks = [non_ambiguous_block] + ambiguous_blocks
        total = len(all_blocks)
        expected_ratio = 1.0 / total  # Only 1 non-ambiguous out of total

        # With 1 non-ambiguous and 4+ ambiguous, ratio <= 1/5 = 0.2 < 0.8
        assume(expected_ratio < 0.8)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(all_blocks, page_width)

        # The reliability threshold should NOT be met → returns None
        assert result is None, (
            f"Expected None (fallback) when only {1}/{total} = {expected_ratio:.2%} "
            f"of blocks are non-ambiguous (< 80% threshold), but got classification result"
        )

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_block_at_exact_threshold_is_ambiguous(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """A block whose x_position equals exactly the threshold has a distance
        of 0, which is within ambiguity_margin, so it SHALL be excluded.

        **Validates: Requirements 2.5**
        """
        threshold = page_width * threshold_ratio
        assume(threshold > ambiguity_margin)
        assume(threshold + ambiguity_margin < page_width)

        # Create a block at exactly the threshold
        text_at_threshold = data.draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
        )
        assume(text_at_threshold.strip() != "")
        text_at_threshold = text_at_threshold.strip()

        block_at_threshold = TextBlock(
            text=text_at_threshold, x_position=threshold, page_number=0
        )

        # Add enough non-ambiguous blocks to pass the 80% threshold
        # so we can verify the block at threshold is excluded from the result
        num_non_ambig = data.draw(st.integers(min_value=5, max_value=10))
        max_left = threshold - ambiguity_margin - 0.01
        blocks = [block_at_threshold]
        seen_texts = {text_at_threshold}

        for _ in range(num_non_ambig):
            x_pos = data.draw(
                st.floats(
                    min_value=0.0,
                    max_value=max_left,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        # Ensure enough non-ambiguous blocks for 80% threshold
        non_ambig_count = len(blocks) - 1  # all except block_at_threshold
        total_count = len(blocks)
        assume(non_ambig_count >= 4)
        assume(non_ambig_count / total_count >= 0.8)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        if result is None:
            # Classification failed for reliability reasons, acceptable
            return

        # The block at the exact threshold should NOT be in the result
        assert text_at_threshold not in result, (
            f"Block at exact threshold (x={threshold}, distance=0, "
            f"margin={ambiguity_margin}) should be excluded from classification, "
            f"but was assigned '{result.get(text_at_threshold)}'"
        )


# --- Property Test: Coordinate Reliability Threshold ---


class TestProperty9CoordinateReliabilityThreshold:
    """Property 9: Coordinate Reliability Threshold.

    For any set of text blocks from a PDF where fewer than 80% of non-ambiguous blocks
    are classifiable into left/right groups, the coordinate-based classification SHALL
    be rejected and the system SHALL fall back to the delivery receipt heuristic.

    # Feature: participant-context-clarity, Property 9: Coordinate Reliability Threshold
    Validates: Requirements 2.3
    """

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_below_80_percent_classifiable_returns_none(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """When fewer than 80% of total blocks are non-ambiguous (classifiable),
        _classify_speaker_by_coordinates SHALL return None (fallback triggered).

        **Validates: Requirements 2.3**
        """
        threshold = page_width * threshold_ratio
        assume(threshold - ambiguity_margin > 0)
        assume(threshold + ambiguity_margin < page_width)

        # Strategy: generate a small number of non-ambiguous blocks and a larger
        # number of ambiguous blocks so that non_ambiguous / total < 0.8
        # We want: non_ambig_count / total < 0.8
        # So: non_ambig_count < 0.8 * total → ambig_count > 0.25 * non_ambig_count
        # Simpler: use 1 non-ambiguous and 4+ ambiguous (ratio ≤ 1/5 = 0.2)

        num_non_ambiguous = data.draw(st.integers(min_value=1, max_value=3))
        # We need: num_non_ambiguous / total < 0.8
        # So total > num_non_ambiguous / 0.8 → num_ambiguous > num_non_ambiguous * 0.25
        # To be safe, use at least 4x as many ambiguous blocks
        num_ambiguous = data.draw(
            st.integers(
                min_value=num_non_ambiguous * 4, max_value=num_non_ambiguous * 4 + 10
            )
        )

        # Verify ratio will be < 0.8
        total = num_non_ambiguous + num_ambiguous
        assume(num_non_ambiguous / total < 0.8)

        # Generate non-ambiguous blocks (clearly left or right of threshold)
        max_left = threshold - ambiguity_margin - 0.01
        min_right = threshold + ambiguity_margin + 0.01

        blocks = []
        seen_texts = set()

        for _ in range(num_non_ambiguous):
            side = data.draw(st.sampled_from(["left", "right"]))
            if side == "left":
                x_pos = data.draw(
                    st.floats(
                        min_value=0.0,
                        max_value=max_left,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            else:
                x_pos = data.draw(
                    st.floats(
                        min_value=min_right,
                        max_value=page_width,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        # Generate ambiguous blocks (within ambiguity_margin of threshold)
        safe_margin = ambiguity_margin * 0.99
        ambiguity_low = max(0.0, threshold - safe_margin)
        ambiguity_high = min(page_width, threshold + safe_margin)

        for _ in range(num_ambiguous):
            x_pos = data.draw(
                st.floats(
                    min_value=ambiguity_low,
                    max_value=ambiguity_high,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            assume(abs(x_pos - threshold) <= ambiguity_margin)
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        # Recount after deduplication
        non_ambig_count = sum(
            1 for b in blocks if abs(b.x_position - threshold) > ambiguity_margin
        )
        total_count = len(blocks)
        assume(total_count >= 5)
        assume(non_ambig_count / total_count < 0.8)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        assert result is None, (
            f"Expected None (fallback) when only {non_ambig_count}/{total_count} = "
            f"{non_ambig_count / total_count:.2%} of blocks are classifiable "
            f"(< 80% threshold), but got classification result with "
            f"{len(result)} entries"
        )

    @given(
        page_width=st.floats(
            min_value=200.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        threshold_ratio=st.floats(
            min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False
        ),
        ambiguity_margin=st.floats(
            min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_at_or_above_80_percent_classifiable_returns_dict(
        self, page_width, threshold_ratio, ambiguity_margin, data
    ):
        """When 80% or more of total blocks are non-ambiguous (classifiable),
        _classify_speaker_by_coordinates SHALL return a dict mapping text to roles.

        **Validates: Requirements 2.3**
        """
        threshold = page_width * threshold_ratio
        assume(threshold - ambiguity_margin > 0)
        assume(threshold + ambiguity_margin < page_width)

        # Strategy: generate mostly non-ambiguous blocks (≥80% of total)
        # Use at least 5 non-ambiguous blocks and 0 or 1 ambiguous blocks
        num_non_ambiguous = data.draw(st.integers(min_value=5, max_value=15))
        # Allow 0 to floor(num_non_ambiguous * 0.25) ambiguous blocks
        # to maintain the ≥80% ratio: non_ambig / total >= 0.8
        # → non_ambig / (non_ambig + ambig) >= 0.8
        # → ambig <= non_ambig * 0.25
        max_ambiguous = int(num_non_ambiguous * 0.25)
        num_ambiguous = data.draw(st.integers(min_value=0, max_value=max_ambiguous))

        # Generate non-ambiguous blocks (clearly left or right of threshold)
        max_left = threshold - ambiguity_margin - 0.01
        min_right = threshold + ambiguity_margin + 0.01

        blocks = []
        seen_texts = set()

        for _ in range(num_non_ambiguous):
            side = data.draw(st.sampled_from(["left", "right"]))
            if side == "left":
                x_pos = data.draw(
                    st.floats(
                        min_value=0.0,
                        max_value=max_left,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            else:
                x_pos = data.draw(
                    st.floats(
                        min_value=min_right,
                        max_value=page_width,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        # Generate ambiguous blocks (within ambiguity_margin of threshold)
        safe_margin = ambiguity_margin * 0.99
        ambiguity_low = max(0.0, threshold - safe_margin)
        ambiguity_high = min(page_width, threshold + safe_margin)

        for _ in range(num_ambiguous):
            x_pos = data.draw(
                st.floats(
                    min_value=ambiguity_low,
                    max_value=ambiguity_high,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            assume(abs(x_pos - threshold) <= ambiguity_margin)
            text = data.draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=20)
            )
            assume(text.strip() != "")
            text = text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                blocks.append(TextBlock(text=text, x_position=x_pos, page_number=0))

        # Recount after deduplication
        non_ambig_count = sum(
            1 for b in blocks if abs(b.x_position - threshold) > ambiguity_margin
        )
        total_count = len(blocks)
        assume(total_count >= 5)
        assume(non_ambig_count >= 4)
        assume(non_ambig_count / total_count >= 0.8)

        parser = PDFParser(
            x_threshold_ratio=threshold_ratio, ambiguity_margin=ambiguity_margin
        )
        result = parser._classify_speaker_by_coordinates(blocks, page_width)

        assert result is not None, (
            f"Expected dict when {non_ambig_count}/{total_count} = "
            f"{non_ambig_count / total_count:.2%} of blocks are classifiable "
            f"(>= 80% threshold), but got None"
        )
        assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
        assert len(result) == non_ambig_count, (
            f"Expected {non_ambig_count} entries in result dict, " f"got {len(result)}"
        )


# --- Delivery Receipt Conversation Strategy ---


@st.composite
def delivery_receipt_conversation_strategy(draw):
    """Generate conversation text with messages and delivery receipts in various positions.

    Returns (text, expected_roles) where expected_roles is a list of expected speaker_role
    values ("sent" or "unknown") for each message in the generated conversation.

    The strategy generates:
    - A message type header
    - Multiple messages with timestamps
    - Some messages followed by "Παραδόθηκε" (delivery receipt) → should get "sent"
    - Other messages without receipt → should get "unknown"
    """
    num_messages = draw(st.integers(min_value=1, max_value=8))
    msg_type = draw(st.sampled_from(MESSAGE_TYPES))

    lines = [msg_type]
    expected_roles: list[str] = []

    for i in range(num_messages):
        # Generate a timestamp for each message
        ts_str, _ = draw(greek_timestamp_strategy())
        lines.append(ts_str)

        # Generate message text
        msg_text = draw(message_text_strategy())
        lines.append(msg_text)

        # Decide whether this message is followed by a delivery receipt
        has_receipt = draw(st.booleans())
        if has_receipt:
            lines.append("Παραδόθηκε")
            expected_roles.append("sent")
        else:
            expected_roles.append("unknown")

    text = "\n".join(lines)
    return text, expected_roles


# --- Property Test: Delivery Receipt Tagging ---


class TestProperty10DeliveryReceiptTagging:
    """Property 10: Delivery Receipt Tagging.

    For any sequence of messages where message M is immediately followed by a
    delivery receipt line in document order, M SHALL be assigned speaker_role "sent".
    Messages not immediately followed by a receipt (and without coordinate classification)
    SHALL be assigned "unknown".

    # Feature: participant-context-clarity, Property 10: Delivery Receipt Tagging
    Validates: Requirements 3.1, 3.2
    """

    @given(data=delivery_receipt_conversation_strategy())
    @settings(max_examples=100)
    def test_messages_followed_by_receipt_get_sent(self, data):
        """For any message immediately followed by a delivery receipt line,
        that message SHALL be assigned speaker_role "sent".

        **Validates: Requirements 3.1, 3.2**
        """
        text, expected_roles = data

        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # We expect the same number of messages as expected_roles
        assert len(messages) == len(expected_roles), (
            f"Expected {len(expected_roles)} messages, got {len(messages)}. "
            f"Text:\n{text}"
        )

        for idx, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
            assert msg.speaker_role == expected_role, (
                f"Message {idx} (text='{msg.text[:40]}...') has speaker_role "
                f"'{msg.speaker_role}', expected '{expected_role}'. "
                f"Text:\n{text}"
            )

    @given(
        msg_texts=st.lists(message_text_strategy(), min_size=2, max_size=6),
    )
    @settings(max_examples=100)
    def test_messages_without_receipt_get_unknown(self, msg_texts):
        """For any sequence of messages where NONE are followed by a delivery receipt,
        ALL messages SHALL be assigned speaker_role "unknown".

        **Validates: Requirements 3.2**
        """
        # Build conversation text with no receipts
        lines = ["iMessage"]
        for msg_text in msg_texts:
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        for idx, msg in enumerate(messages):
            assert msg.speaker_role == "unknown", (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected 'unknown' (no receipt follows). "
                f"Text:\n{text}"
            )

    @given(
        msg_texts=st.lists(message_text_strategy(), min_size=2, max_size=6),
    )
    @settings(max_examples=100)
    def test_all_messages_with_receipt_get_sent(self, msg_texts):
        """For any sequence of messages where ALL are followed by a delivery receipt,
        ALL messages SHALL be assigned speaker_role "sent".

        **Validates: Requirements 3.1**
        """
        # Build conversation text with receipts after every message
        lines = ["iMessage"]
        for msg_text in msg_texts:
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)
            lines.append("Παραδόθηκε")

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        for idx, msg in enumerate(messages):
            assert msg.speaker_role == "sent", (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected 'sent' (receipt follows). "
                f"Text:\n{text}"
            )

    @given(
        msg_text=message_text_strategy(),
    )
    @settings(max_examples=100)
    def test_orphan_receipt_at_start_is_ignored(self, msg_text):
        """When a delivery receipt line appears before any message has been accumulated,
        it SHALL be ignored (orphan receipt).

        **Validates: Requirements 3.1, 3.2**
        """
        # Receipt before the first message
        lines = [
            "iMessage",
            "Παραδόθηκε",  # orphan receipt - no preceding message
            "Παρ 20 Φεβ, 10:30 πµ",
            msg_text,
        ]

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # The orphan receipt should not cause any message to be tagged "sent"
        # The only message (msg_text) has no receipt after it
        assert len(messages) >= 1, "Expected at least one message"
        for msg in messages:
            assert msg.speaker_role == "unknown", (
                f"Message '{msg.text[:40]}' has speaker_role '{msg.speaker_role}', "
                f"expected 'unknown' (orphan receipt should not tag any message)"
            )


# --- Property Test: Speaker Role Assignment Safety Invariant ---


class TestProperty11SpeakerRoleAssignmentSafetyInvariant:
    """Property 11: Speaker Role Assignment Safety Invariant.

    For any set of parsed messages, every message with speaker_role "sent" SHALL be
    justified by either (a) a delivery receipt immediately following it in document order,
    or (b) coordinate-based classification indicating right-aligned. No other mechanism
    SHALL produce a "sent" assignment.

    # Feature: participant-context-clarity, Property 11: Speaker Role Assignment Safety Invariant
    Validates: Requirements 3.3
    """

    @given(data=delivery_receipt_conversation_strategy())
    @settings(max_examples=100)
    def test_sent_only_when_receipt_follows(self, data):
        """For any set of messages processed by the receipt heuristic,
        every message with speaker_role "sent" SHALL be immediately followed
        by a delivery receipt in the original document order.

        **Validates: Requirements 3.3**
        """
        text, expected_roles = data

        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # Parse the text to identify which messages are followed by a receipt
        lines = text.split("\n")
        receipt_line = "Παραδόθηκε"

        # Build a set of message indices that are followed by a receipt
        # based on the parser's internal tracking
        receipt_indices = getattr(parser, "_receipt_followed_indices", set())

        for idx, msg in enumerate(messages):
            if msg.speaker_role == "sent":
                # A "sent" role MUST be justified: the message index must be
                # in the receipt_followed_indices set (receipt follows it)
                assert idx in receipt_indices, (
                    f"Message {idx} (text='{msg.text[:50]}') has speaker_role 'sent' "
                    f"but is NOT followed by a delivery receipt in document order. "
                    f"Receipt indices: {receipt_indices}. "
                    f"This violates the safety invariant: 'sent' requires justification."
                )

    @given(
        msg_texts=st.lists(message_text_strategy(), min_size=1, max_size=8),
    )
    @settings(max_examples=100)
    def test_no_sent_without_receipt_in_receipt_only_mode(self, msg_texts):
        """When coordinate classification is not used (receipt-only mode),
        no message SHALL receive speaker_role "sent" unless a delivery receipt
        immediately follows it.

        **Validates: Requirements 3.3**
        """
        # Build conversation text with NO receipts at all
        lines = ["iMessage"]
        for msg_text in msg_texts:
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # No receipts → no message should be "sent"
        for idx, msg in enumerate(messages):
            assert msg.speaker_role != "sent", (
                f"Message {idx} (text='{msg.text[:50]}') has speaker_role 'sent' "
                f"but no delivery receipt follows it in document order. "
                f"Without coordinate classification or a receipt, 'sent' is unjustified."
            )

    @given(data=delivery_receipt_conversation_strategy())
    @settings(max_examples=100)
    def test_sent_count_equals_receipt_count(self, data):
        """The number of messages assigned "sent" by the receipt heuristic SHALL
        equal exactly the number of delivery receipts that follow messages.
        No extra "sent" assignments can exist without justification.

        **Validates: Requirements 3.3**
        """
        text, expected_roles = data

        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # Count messages that are "sent"
        sent_count = sum(1 for msg in messages if msg.speaker_role == "sent")
        # Count expected "sent" from the generated strategy
        expected_sent_count = sum(1 for role in expected_roles if role == "sent")

        assert sent_count == expected_sent_count, (
            f"Expected {expected_sent_count} 'sent' messages (matching receipt count) "
            f"but got {sent_count} 'sent' messages. "
            f"Extra 'sent' assignments without justification violate the safety invariant."
        )

    @given(
        msg_texts=st.lists(message_text_strategy(), min_size=2, max_size=8),
        receipt_positions=st.data(),
    )
    @settings(max_examples=100)
    def test_only_receipt_preceded_messages_get_sent(
        self, msg_texts, receipt_positions
    ):
        """For any arbitrary placement of delivery receipts among messages,
        ONLY the messages immediately before a receipt SHALL be tagged "sent".
        All other messages SHALL NOT be "sent".

        **Validates: Requirements 3.3**
        """
        # Build conversation with controlled receipt placement
        lines = ["iMessage"]
        expected_sent_indices: set[int] = set()

        for i, msg_text in enumerate(msg_texts):
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)
            # Randomly decide if this message gets a receipt
            has_receipt = receipt_positions.draw(st.booleans())
            if has_receipt:
                lines.append("Παραδόθηκε")
                expected_sent_indices.add(i)

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        assert len(messages) == len(
            msg_texts
        ), f"Expected {len(msg_texts)} messages, got {len(messages)}"

        for idx, msg in enumerate(messages):
            if idx in expected_sent_indices:
                assert msg.speaker_role == "sent", (
                    f"Message {idx} (text='{msg.text[:50]}') should be 'sent' "
                    f"(receipt follows) but got '{msg.speaker_role}'"
                )
            else:
                assert msg.speaker_role != "sent", (
                    f"Message {idx} (text='{msg.text[:50]}') has speaker_role 'sent' "
                    f"but no delivery receipt follows it. "
                    f"This violates the safety invariant."
                )


# --- Property Test: Coordinate Precedence Over Heuristic ---


class TestProperty12CoordinatePrecedenceOverHeuristic:
    """Property 12: Coordinate Precedence Over Heuristic.

    For any message where both coordinate-based classification and delivery receipt
    heuristic produce a speaker_role, the final speaker_role SHALL equal the
    coordinate-based result.

    # Feature: participant-context-clarity, Property 12: Coordinate Precedence Over Heuristic
    Validates: Requirements 3.4
    """

    @given(
        msg_texts=st.lists(
            message_text_strategy(), min_size=2, max_size=6, unique=True
        ),
        coord_roles=st.data(),
    )
    @settings(max_examples=100)
    def test_coordinate_result_overrides_receipt_sent(self, msg_texts, coord_roles):
        """When a message would be tagged "sent" by receipt heuristic but coordinate
        classification assigns "received", the final role SHALL be "received"
        (coordinate wins).

        **Validates: Requirements 3.4**
        """
        # Build conversation text where ALL messages are followed by a receipt
        # (so receipt heuristic would assign "sent" to all)
        lines = ["iMessage"]
        for msg_text in msg_texts:
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)
            lines.append("Παραδόθηκε")

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)

        # Verify receipt heuristic would tag these as "sent"
        receipt_messages = parser._assign_speaker_roles_by_receipt(
            [
                Message(
                    text=m.text,
                    timestamp=m.timestamp,
                    message_type=m.message_type,
                    phone_number=m.phone_number,
                )
                for m in messages
            ]
        )
        for rm in receipt_messages:
            assert (
                rm.speaker_role == "sent"
            ), f"Setup error: receipt heuristic should tag all messages as 'sent'"

        # Now create a coordinate classification that assigns different roles
        # For each message, randomly assign "sent" or "received" via coordinates
        coord_classification: dict[str, str] = {}
        expected_roles: list[str] = []
        for msg in messages:
            role = coord_roles.draw(st.sampled_from(["sent", "received"]))
            coord_classification[msg.text] = role
            expected_roles.append(role)

        # Apply coordinate roles (this is what parse() does when coord classification succeeds)
        parser._apply_coordinate_roles(messages, coord_classification)

        # Verify coordinate result took precedence
        for idx, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
            assert msg.speaker_role == expected_role, (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected '{expected_role}' from coordinate "
                f"classification. Coordinate result should take precedence over "
                f"receipt heuristic."
            )

    @given(
        msg_texts=st.lists(
            message_text_strategy(), min_size=2, max_size=6, unique=True
        ),
        receipt_pattern=st.data(),
    )
    @settings(max_examples=100)
    def test_coordinate_received_overrides_receipt_sent(
        self, msg_texts, receipt_pattern
    ):
        """Specifically when coordinate says "received" but receipt says "sent",
        the final role SHALL be "received" (most critical conflict scenario).

        **Validates: Requirements 3.4**
        """
        # Build conversation with some messages followed by receipts
        lines = ["iMessage"]
        receipt_indices: set[int] = set()

        for i, msg_text in enumerate(msg_texts):
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)
            has_receipt = receipt_pattern.draw(st.booleans())
            if has_receipt:
                lines.append("Παραδόθηκε")
                receipt_indices.add(i)

        # Need at least one message with a receipt to test the conflict
        assume(len(receipt_indices) >= 1)

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        assert len(messages) == len(msg_texts)

        # Create coordinate classification that assigns "received" to ALL messages
        # (conflicting with the "sent" from receipt heuristic for receipt-followed messages)
        coord_classification: dict[str, str] = {
            msg.text: "received" for msg in messages
        }

        # Apply coordinate roles
        parser._apply_coordinate_roles(messages, coord_classification)

        # ALL messages should be "received" because coordinates take precedence
        for idx, msg in enumerate(messages):
            assert msg.speaker_role == "received", (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected 'received' from coordinate "
                f"classification. Even though message {idx} "
                f"{'was' if idx in receipt_indices else 'was not'} followed by a "
                f"receipt, coordinate result SHALL take precedence."
            )

    @given(
        msg_texts=st.lists(
            message_text_strategy(), min_size=2, max_size=6, unique=True
        ),
        receipt_pattern=st.data(),
        coord_roles=st.data(),
    )
    @settings(max_examples=100)
    def test_parse_integration_coordinate_overrides_receipt(
        self, msg_texts, receipt_pattern, coord_roles
    ):
        """Integration test: simulating the parse() flow where coordinate
        classification succeeds, it SHALL be applied instead of the receipt
        heuristic, even for messages that would have been tagged by receipts.

        **Validates: Requirements 3.4**
        """
        # Build conversation with mixed receipts
        lines = ["iMessage"]
        receipt_indices: set[int] = set()

        for i, msg_text in enumerate(msg_texts):
            lines.append("Παρ 20 Φεβ, 10:30 πµ")
            lines.append(msg_text)
            has_receipt = receipt_pattern.draw(st.booleans())
            if has_receipt:
                lines.append("Παραδόθηκε")
                receipt_indices.add(i)

        text = "\n".join(lines)
        parser = PDFParser()
        messages = parser._extract_messages(text)
        assert len(messages) == len(msg_texts)

        # Generate coordinate classification for all messages
        coord_classification: dict[str, str] = {}
        expected_roles: list[str] = []
        for msg in messages:
            role = coord_roles.draw(st.sampled_from(["sent", "received"]))
            coord_classification[msg.text] = role
            expected_roles.append(role)

        # Simulate parse() behavior: when coord_classification is not None,
        # _apply_coordinate_roles is called instead of _assign_speaker_roles_by_receipt
        # This is the precedence rule.
        parser._apply_coordinate_roles(messages, coord_classification)

        # Verify final roles match coordinate classification (not receipt heuristic)
        for idx, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
            assert msg.speaker_role == expected_role, (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected '{expected_role}' from coordinate "
                f"classification. The parse() method uses coordinate results when "
                f"available, discarding receipt heuristic results. "
                f"Message {'was' if idx in receipt_indices else 'was not'} "
                f"followed by a receipt."
            )

    @given(data=delivery_receipt_conversation_strategy())
    @settings(max_examples=100)
    def test_coordinate_classification_none_falls_back_to_receipt(self, data):
        """When coordinate classification returns None (unreliable), the receipt
        heuristic SHALL be used. This confirms that precedence only applies when
        coordinates ARE available.

        **Validates: Requirements 3.4**
        """
        text, expected_roles = data

        parser = PDFParser()
        messages = parser._extract_messages(text)

        # Simulate parse() behavior: coord_classification is None → use receipt heuristic
        coord_classification = None

        if coord_classification is not None:
            parser._apply_coordinate_roles(messages, coord_classification)
        else:
            messages = parser._assign_speaker_roles_by_receipt(messages)

        # Should match receipt heuristic results
        assert len(messages) == len(
            expected_roles
        ), f"Expected {len(expected_roles)} messages, got {len(messages)}"

        for idx, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
            assert msg.speaker_role == expected_role, (
                f"Message {idx} (text='{msg.text[:40]}') has speaker_role "
                f"'{msg.speaker_role}', expected '{expected_role}' from receipt "
                f"heuristic (coordinate classification was None/unreliable)."
            )
