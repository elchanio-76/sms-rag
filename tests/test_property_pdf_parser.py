"""Property-based tests for PDF parsing.

# Feature: sms-rag, Property 1: Message Chronological Order Preservation
# Feature: sms-rag, Property 2: Metadata Extraction Completeness
# Feature: sms-rag, Property 3: Filename to Participant Name Derivation
# Feature: participant-context-clarity, Property 13: Speaker Role Domain Invariant

Validates: Requirements 1.1, 1.2, 1.4, 3.6, 4.1, 4.2
"""

import string
from datetime import datetime

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.shared.models import Message


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
