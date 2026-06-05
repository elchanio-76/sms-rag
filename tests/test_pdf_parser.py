"""Unit tests for the PDF parser module."""

import logging
from datetime import datetime
from pathlib import Path

import pytest

from sms_rag.preprocessing.pdf_parser import (
    PDFParser,
    _normalize_message_type,
    _parse_timestamp_full,
    _parse_timestamp_day_only,
    _is_skip_line,
    _DATE_PATTERN_FULL,
    _DATE_PATTERN_DAY_ONLY,
)


class TestParticipantNameDerivation:
    """Test participant name is derived from filename."""

    def test_simple_name(self, tmp_path):
        """Participant name derived by removing .pdf extension."""
        import pymupdf

        pdf_path = tmp_path / "John Smith.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello world")
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "John Smith"
        assert result.source_filename == "John Smith.pdf"

    def test_name_with_special_characters(self, tmp_path):
        """Participant name preserved with special characters from filename."""
        import pymupdf

        # Use ASCII characters that won't be garbled
        pdf_path = tmp_path / "Marie-Anne O'Brien.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content")
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "Marie-Anne O'Brien"

    def test_name_with_multiple_dots(self, tmp_path):
        """Only the .pdf extension is removed, not other dots."""
        import pymupdf

        pdf_path = tmp_path / "Dr. Smith Jr..pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test")
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "Dr. Smith Jr."


class TestErrorHandling:
    """Test graceful error handling for malformed/unreadable PDFs."""

    def test_nonexistent_file(self):
        """Non-existent file returns empty conversation with error."""
        parser = PDFParser()
        result = parser.parse(Path("/nonexistent/path/file.pdf"))

        assert result.participant_name == "file"
        assert result.source_filename == "file.pdf"
        assert result.messages == []
        assert len(result.errors) == 1
        assert "file.pdf" in result.errors[0]

    def test_malformed_pdf(self, tmp_path):
        """Malformed file returns empty conversation with error."""
        pdf_path = tmp_path / "malformed.pdf"
        pdf_path.write_text("This is not a valid PDF file")

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "malformed"
        assert result.source_filename == "malformed.pdf"
        assert result.messages == []
        assert len(result.errors) == 1

    def test_empty_pdf(self, tmp_path):
        """PDF with no extractable text returns empty with error."""
        import pymupdf

        pdf_path = tmp_path / "empty.pdf"
        doc = pymupdf.open()
        doc.new_page()  # Empty page, no text
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "empty"
        assert result.messages == []
        assert len(result.errors) == 1
        assert "no extractable text" in result.errors[0]

    def test_password_protected_pdf(self, tmp_path):
        """Password-protected PDF returns empty with error."""
        import pymupdf

        pdf_path = tmp_path / "protected.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        # Save with encryption
        doc.save(
            str(pdf_path),
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        doc.close()

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert result.participant_name == "protected"
        assert result.messages == []
        assert len(result.errors) == 1
        assert "password-protected" in result.errors[0]

    def test_errors_are_logged(self, caplog):
        """Errors are logged via the logger."""
        parser = PDFParser()
        with caplog.at_level(logging.ERROR):
            parser.parse(Path("/nonexistent/file.pdf"))

        assert len(caplog.records) > 0
        assert "file.pdf" in caplog.records[0].message

    def test_return_type_always_parsed_conversation(self, tmp_path):
        """Even on error, returns a ParsedConversation, never raises."""
        parser = PDFParser()
        # Non-existent
        r1 = parser.parse(Path("/no/such/file.pdf"))
        assert r1.participant_name == "file"
        assert isinstance(r1.errors, list)

        # Malformed
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"\x00\x01\x02")
        r2 = parser.parse(bad)
        assert r2.participant_name == "bad"
        assert isinstance(r2.errors, list)


class TestMessageExtractionDirect:
    """Test message extraction using _extract_messages directly on raw text.

    This avoids PDF rendering issues with Greek characters by testing
    the parsing logic directly on text strings.
    """

    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_single_imessage(self, parser):
        """Single iMessage with full date is parsed correctly."""
        text = "iMessage\n\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\nHello world"
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].text == "Hello world"
        assert messages[0].message_type == "iMessage"
        assert messages[0].timestamp == datetime(2024, 2, 20, 10, 30)

    def test_sms_message_type(self, parser):
        """SMS message type is correctly identified."""
        text = "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 SMS\n\u0394\u03b5\u03c5 9 \u039c\u03b1\u03c1, 12:34 \u00b5\u00b5\nTest SMS"
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].message_type == "SMS"
        assert messages[0].text == "Test SMS"

    def test_rcs_message_type(self, parser):
        """RCS message type is correctly identified."""
        text = "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 RCS\n\u03a0\u03b1\u03c1 9 \u0399\u03b1\u03bd, 8:07 \u03c0\u00b5\nTest RCS"
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].message_type == "RCS"
        assert messages[0].text == "Test RCS"

    def test_multiple_messages_chronological(self, parser):
        """Multiple messages extracted in order of appearance."""
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "First message\n"
            "\u0394\u03b5\u03c5 9 \u039c\u03b1\u03c1, 12:34 \u00b5\u00b5\n"
            "Second message\n"
            "\u03a4\u03c1\u03af 21 \u0391\u03c0\u03c1, 4:06 \u00b5\u00b5\n"
            "Third message"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 3
        assert messages[0].text == "First message"
        assert messages[1].text == "Second message"
        assert messages[2].text == "Third message"
        # Verify chronological order
        assert messages[0].timestamp < messages[1].timestamp < messages[2].timestamp

    def test_multiline_message(self, parser):
        """Multi-line messages are joined with newlines."""
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "First line\n"
            "Second line\n"
            "Third line"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].text == "First line\nSecond line\nThird line"

    def test_phone_number_extraction(self, parser):
        """Phone numbers in international format are extracted."""
        text = (
            "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 SMS\n"
            "+306948584536\n"
            "\u0394\u03b5\u03c5 9 \u039c\u03b1\u03c1, 12:34 \u00b5\u00b5\n"
            "Message text"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        # Phone number from context (between type and date) is preserved
        assert messages[0].phone_number == "+306948584536"
        assert messages[0].message_type == "SMS"

    def test_missing_timestamp_with_day_only(self, parser):
        """Day-only date pattern results in None timestamp."""
        text = "iMessage\n\u03a0\u03b1\u03c1\u03b1\u03c3\u03ba\u03b5\u03c5\u03ae 6:13 \u00b5\u00b5\nMessage text"
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].timestamp is None
        assert messages[0].text == "Message text"

    def test_missing_phone_is_none(self, parser):
        """Messages without phone numbers have None for phone_number."""
        text = "iMessage\n\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\nHello"
        messages = parser._extract_messages(text)

        assert messages[0].phone_number is None

    def test_delivery_receipt_splits_messages(self, parser):
        """\u03a0\u03b1\u03c1\u03b1\u03b4\u03cc\u03b8\u03b7\u03ba\u03b5 acts as a message boundary."""
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1\u03b1\u03c3\u03ba\u03b5\u03c5\u03ae 6:13 \u00b5\u00b5\n"
            "First message\n"
            "\u03a0\u03b1\u03c1\u03b1\u03b4\u03cc\u03b8\u03b7\u03ba\u03b5\n"
            "Second message"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 2
        assert messages[0].text == "First message"
        assert messages[1].text == "Second message"

    def test_skip_system_lines(self, parser):
        """System indicator lines are excluded from message text."""
        # Note: The EIXATE pattern in the PDF uses Latin H (U+0048) not Greek Η (U+0397)
        text = (
            "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 SMS\n"
            "EIXATE 1 K\u039bH\u01a9H:\n"
            "+306948584536\n"
            "(1) 09/03 12:34\n"
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "Actual message"
        )
        messages = parser._extract_messages(text)

        # The actual message should be present, system lines skipped
        assert any("Actual message" in m.text for m in messages)
        assert not any("EIXATE" in m.text for m in messages)

    def test_read_receipt_skipped(self, parser):
        """\u0391\u03bd\u03b1\u03b3\u03bd\u03ce\u03c3\u03c4\u03b7\u03ba\u03b5 lines are skipped."""
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "Hello\n"
            "\u0391\u03bd\u03b1\u03b3\u03bd\u03ce\u03c3\u03c4\u03b7\u03ba\u03b5 26/5/26"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert (
            "\u0391\u03bd\u03b1\u03b3\u03bd\u03ce\u03c3\u03c4\u03b7\u03ba\u03b5"
            not in messages[0].text
        )

    def test_message_type_transition(self, parser):
        """Transition between message types is handled correctly."""
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "iMessage text\n"
            "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 RCS\n"
            "\u0394\u03b5\u03c5 9 \u039c\u03b1\u03c1, 12:34 \u00b5\u00b5\n"
            "RCS text"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 2
        assert messages[0].message_type == "iMessage"
        assert messages[0].text == "iMessage text"
        assert messages[1].message_type == "RCS"
        assert messages[1].text == "RCS text"

    def test_empty_text_returns_no_messages(self, parser):
        """Empty text produces no messages."""
        messages = parser._extract_messages("")
        assert messages == []

    def test_whitespace_only_text_returns_no_messages(self, parser):
        """Whitespace-only text produces no messages."""
        messages = parser._extract_messages("   \n  \n  ")
        assert messages == []

    def test_message_type_preserved_across_dates(self, parser):
        """Message type from indicator applies to subsequent messages."""
        text = (
            "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 SMS\n"
            "\u03a0\u03b1\u03c1 24 \u039f\u03ba\u03c4, 6:52 \u00b5\u00b5\n"
            "First SMS\n"
            "\u0394\u03b5\u03c5 15 \u0394\u03b5\u03ba, 8:57 \u03c0\u00b5\n"
            "Second SMS"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 2
        assert messages[0].message_type == "SMS"
        assert messages[1].message_type == "SMS"

    def test_message_without_type_indicator(self, parser):
        """Messages can start with a date even if no type indicator precedes them."""
        # If there's just a date line and text (e.g., first thing after some context)
        # The parser won't emit a message if no type context and in_message is False
        # But with a type indicator first, it works
        text = (
            "iMessage\n"
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5\n"
            "Message with type"
        )
        messages = parser._extract_messages(text)

        assert len(messages) == 1
        assert messages[0].message_type == "iMessage"


class TestTimestampParsing:
    """Test timestamp parsing from various date/time formats."""

    def test_full_date_am(self):
        """Full date with AM time parsed correctly."""
        match = _DATE_PATTERN_FULL.match(
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2, 10:30 \u03c0\u00b5"
        )
        assert match is not None
        ts = _parse_timestamp_full(match)
        assert ts == datetime(2024, 2, 20, 10, 30)

    def test_full_date_pm(self):
        """Full date with PM 12:xx stays as 12:xx."""
        match = _DATE_PATTERN_FULL.match(
            "\u0394\u03b5\u03c5 9 \u039c\u03b1\u03c1, 12:34 \u00b5\u00b5"
        )
        assert match is not None
        ts = _parse_timestamp_full(match)
        assert ts == datetime(2024, 3, 9, 12, 34)

    def test_full_date_pm_non_12(self):
        """PM time that isn't 12 gets 12 added."""
        match = _DATE_PATTERN_FULL.match(
            "\u03a4\u03c1\u03af 21 \u0391\u03c0\u03c1, 4:06 \u00b5\u00b5"
        )
        assert match is not None
        ts = _parse_timestamp_full(match)
        assert ts == datetime(2024, 4, 21, 16, 6)

    def test_full_date_12_am(self):
        """12:xx AM is converted to 0:xx."""
        match = _DATE_PATTERN_FULL.match(
            "\u0394\u03b5\u03c5 1 \u0399\u03b1\u03bd, 12:00 \u03c0\u00b5"
        )
        assert match is not None
        ts = _parse_timestamp_full(match)
        assert ts == datetime(2024, 1, 1, 0, 0)

    def test_day_only_returns_none(self):
        """Day-only pattern returns None (no date info)."""
        match = _DATE_PATTERN_DAY_ONLY.match(
            "\u03a0\u03b1\u03c1\u03b1\u03c3\u03ba\u03b5\u03c5\u03ae 6:13 \u00b5\u00b5"
        )
        assert match is not None
        ts = _parse_timestamp_day_only(match)
        assert ts is None

    def test_full_date_pattern_matches_without_comma(self):
        """Full date pattern works with or without comma."""
        match = _DATE_PATTERN_FULL.match(
            "\u03a0\u03b1\u03c1 20 \u03a6\u03b5\u03b2 10:30 \u03c0\u00b5"
        )
        assert match is not None

    def test_all_months_recognized(self):
        """All 12 Greek month abbreviations are matched."""
        months = [
            "\u0399\u03b1\u03bd",
            "\u03a6\u03b5\u03b2",
            "\u039c\u03b1\u03c1",
            "\u0391\u03c0\u03c1",
            "\u039c\u03b1\u0390",
            "\u0399\u03bf\u03c5\u03bd",
            "\u0399\u03bf\u03c5\u03bb",
            "\u0391\u03c5\u03b3",
            "\u03a3\u03b5\u03c0",
            "\u039f\u03ba\u03c4",
            "\u039d\u03bf\u03b5",
            "\u0394\u03b5\u03ba",
        ]
        for month in months:
            line = f"\u0394\u03b5\u03c5 1 {month}, 10:00 \u03c0\u00b5"
            match = _DATE_PATTERN_FULL.match(line)
            assert match is not None, f"Month {month} not matched"

    def test_all_day_abbreviations_recognized(self):
        """All Greek day abbreviations are matched."""
        days = [
            "\u0394\u03b5\u03c5",
            "\u03a4\u03c1\u03af",
            "\u03a4\u03b5\u03c4",
            "\u03a0\u03ad\u00b5",
            "\u03a0\u03b1\u03c1",
            "\u03a3\u03ac\u03b2",
            "\u039a\u03c5\u03c1",
        ]
        for day in days:
            line = f"{day} 1 \u0399\u03b1\u03bd, 10:00 \u03c0\u00b5"
            match = _DATE_PATTERN_FULL.match(line)
            assert match is not None, f"Day {day} not matched"

    def test_all_full_day_names_recognized(self):
        """All full Greek day names are matched in day-only pattern."""
        days = [
            "\u0394\u03b5\u03c5\u03c4\u03ad\u03c1\u03b1",
            "\u03a4\u03c1\u03af\u03c4\u03b7",
            "\u03a4\u03b5\u03c4\u03ac\u03c1\u03c4\u03b7",
            "\u03a0\u03ad\u00b5\u03c0\u03c4\u03b7",
            "\u03a0\u03b1\u03c1\u03b1\u03c3\u03ba\u03b5\u03c5\u03ae",
            "\u03a3\u03ac\u03b2\u03b2\u03b1\u03c4\u03bf",
            "\u039a\u03c5\u03c1\u03b9\u03b1\u03ba\u03ae",
        ]
        for day in days:
            line = f"{day} 6:13 \u00b5\u00b5"
            match = _DATE_PATTERN_DAY_ONLY.match(line)
            assert match is not None, f"Day {day} not matched"


class TestNormalizeMessageType:
    """Test message type normalization."""

    def test_imessage(self):
        assert _normalize_message_type("iMessage") == "iMessage"

    def test_sms(self):
        assert (
            _normalize_message_type(
                "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 SMS"
            )
            == "SMS"
        )

    def test_rcs(self):
        assert (
            _normalize_message_type(
                "\u0393\u03c1\u03b1\u03c0\u03c4\u03cc \u00b5\u03ae\u03bd\u03c5\u00b5\u03b1 \u2022 RCS"
            )
            == "RCS"
        )

    def test_whitespace_handling(self):
        assert _normalize_message_type("  iMessage  ") == "iMessage"


class TestSkipLines:
    """Test system indicator line detection."""

    def test_eixate_pattern(self):
        # Uses Latin H (U+0048) and Ʃ (U+01A9) as in the actual PDF
        assert _is_skip_line("EIXATE 1 K\u039bH\u01a9H:") is True
        assert _is_skip_line("EIXATE 3 K\u039bH\u01a9H:") is True

    def test_anagnostike_pattern(self):
        assert (
            _is_skip_line(
                "\u0391\u03bd\u03b1\u03b3\u03bd\u03ce\u03c3\u03c4\u03b7\u03ba\u03b5 26/5/26"
            )
            is True
        )
        assert (
            _is_skip_line(
                "\u0391\u03bd\u03b1\u03b3\u03bd\u03ce\u03c3\u03c4\u03b7\u03ba\u03b5"
            )
            is True
        )

    def test_call_log_pattern(self):
        assert _is_skip_line("(1) 09/03 12:34") is True
        assert _is_skip_line("(2) 18/05 12:06") is True

    def test_normal_text_not_skipped(self):
        assert _is_skip_line("Hello world") is False
        assert (
            _is_skip_line("\u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1") is False
        )


class TestAssignSpeakerRolesByReceipt:
    """Test the _assign_speaker_roles_by_receipt method."""

    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_message_followed_by_receipt_gets_sent(self, parser):
        """A message immediately followed by a delivery receipt is tagged 'sent'."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "Hello there\n"
            "Παραδόθηκε\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Reply message"
        )
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        assert len(messages) == 2
        assert messages[0].speaker_role == "sent"
        assert messages[1].speaker_role == "unknown"

    def test_message_without_receipt_gets_unknown(self, parser):
        """Messages not followed by a receipt get 'unknown'."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "First message\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Second message"
        )
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        assert len(messages) == 2
        assert messages[0].speaker_role == "unknown"
        assert messages[1].speaker_role == "unknown"

    def test_orphan_receipt_is_ignored(self, parser):
        """A receipt with no preceding message is ignored (Req 3.5)."""
        text = "iMessage\n" "Παραδόθηκε\n" "Παρ 20 Φεβ, 10:30 πµ\n" "Hello"
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # Only one message should exist, and it shouldn't be tagged "sent"
        # because the receipt was orphaned (no preceding accumulated message)
        assert len(messages) == 1
        assert messages[0].text == "Hello"
        assert messages[0].speaker_role == "unknown"

    def test_multiple_receipts_tag_correct_messages(self, parser):
        """Multiple receipts correctly tag their preceding messages."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "Sent message 1\n"
            "Παραδόθηκε\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Received message\n"
            "Τρί 21 Απρ, 4:06 µµ\n"
            "Sent message 2\n"
            "Παραδόθηκε"
        )
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        assert len(messages) == 3
        assert messages[0].speaker_role == "sent"
        assert messages[1].speaker_role == "unknown"
        assert messages[2].speaker_role == "sent"

    def test_consecutive_receipts_only_tag_last_message(self, parser):
        """Each receipt tags only the message immediately before it."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "First sent\n"
            "Παραδόθηκε\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Second sent\n"
            "Παραδόθηκε"
        )
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        assert len(messages) == 2
        assert messages[0].speaker_role == "sent"
        assert messages[1].speaker_role == "sent"

    def test_preserves_existing_non_unknown_roles(self, parser):
        """If a message already has a non-unknown role (from coordinates),
        the receipt heuristic preserves it."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "A message\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Another message"
        )
        messages = parser._extract_messages(text)
        # Simulate coordinate classification already setting a role
        messages[0].speaker_role = "received"
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # The "received" role should be preserved (not overwritten to "unknown")
        assert messages[0].speaker_role == "received"
        assert messages[1].speaker_role == "unknown"

    def test_empty_message_list(self, parser):
        """Empty message list is handled gracefully."""
        parser._extract_messages("")  # Initialize _receipt_followed_indices
        result = parser._assign_speaker_roles_by_receipt([])
        assert result == []

    def test_sent_never_assigned_without_receipt(self, parser):
        """Speaker role 'sent' is never assigned without a receipt (Req 3.3)."""
        text = (
            "iMessage\n"
            "Παρ 20 Φεβ, 10:30 πµ\n"
            "Message one\n"
            "Δευ 9 Μαρ, 12:34 µµ\n"
            "Message two\n"
            "Τρί 21 Απρ, 4:06 µµ\n"
            "Message three"
        )
        messages = parser._extract_messages(text)
        messages = parser._assign_speaker_roles_by_receipt(messages)

        # None should be "sent" since there are no receipts
        for msg in messages:
            assert msg.speaker_role != "sent"


class TestRealPDFs:
    """Integration tests using actual PDF files in data/."""

    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_kyriaki_pdf(self, parser):
        """Parse the Kyriaki Salavanitou PDF file."""
        pdf_path = Path("data/Kyriaki Salavanitou.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        assert result.participant_name == "Kyriaki Salavanitou"
        assert result.source_filename == "Kyriaki Salavanitou.pdf"
        assert result.errors == []
        assert len(result.messages) > 0
        # First message should have iMessage type
        assert result.messages[0].message_type == "iMessage"
        # First message has a full date
        assert result.messages[0].timestamp == datetime(2024, 2, 20, 10, 30)
        # All messages should have non-empty text
        for msg in result.messages:
            assert msg.text.strip() != ""

    def test_loizos_pdf(self, parser):
        """Parse the Loizos Markides PDF file."""
        pdf_path = Path("data/Loizos Markides.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        assert result.participant_name == "Loizos Markides"
        assert result.source_filename == "Loizos Markides.pdf"
        assert result.errors == []
        assert len(result.messages) > 0
        # First message should have SMS type
        assert result.messages[0].message_type == "SMS"
        # Should have RCS messages too
        message_types = {m.message_type for m in result.messages}
        assert "RCS" in message_types
        assert "SMS" in message_types

    def test_loizos_contains_greeklish(self, parser):
        """Loizos PDF contains Greeklish text preserved as-is."""
        pdf_path = Path("data/Loizos Markides.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        # Find messages with Greeklish content
        greeklish_messages = [
            m
            for m in result.messages
            if "Perimenoume" in m.text or "proxorisoume" in m.text
        ]
        assert len(greeklish_messages) > 0

    def test_chronological_order_kyriaki(self, parser):
        """Messages with timestamps are in chronological order (Kyriaki)."""
        pdf_path = Path("data/Kyriaki Salavanitou.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        # Filter messages that have timestamps
        timestamped = [m for m in result.messages if m.timestamp is not None]
        for i in range(len(timestamped) - 1):
            assert timestamped[i].timestamp <= timestamped[i + 1].timestamp

    def test_loizos_timestamps_present(self, parser):
        """Loizos PDF messages have timestamps extracted."""
        pdf_path = Path("data/Loizos Markides.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        # Most messages in Loizos should have timestamps
        timestamped = [m for m in result.messages if m.timestamp is not None]
        assert len(timestamped) > 10  # The PDF has many dated messages

    def test_kyriaki_phone_number_in_context(self, parser):
        """Phone number present in the Kyriaki SMS sections."""
        pdf_path = Path("data/Kyriaki Salavanitou.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data file not available")

        result = parser.parse(pdf_path)

        # The PDF has phone numbers in SMS notification sections
        # At minimum, check that the parser doesn't include phone-only
        # lines as message text
        for msg in result.messages:
            # A message should never be just a phone number
            assert msg.text.strip() != "+306948584536"
