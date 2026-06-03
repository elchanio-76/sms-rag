"""Property-based tests for the PreprocessingPipeline.

# Feature: sms-rag, Property 4: Text Preservation Through Pipeline
# Feature: sms-rag, Property 15: Pipeline Idempotence
# Feature: sms-rag, Property 16: Pipeline Resilience and Summary Accuracy

Validates: Requirements 2.2, 2.4, 8.3, 8.4, 8.5

Property 4: For any conversation chunk containing Greek, English, or Greeklish
text, the text stored in the vector store SHALL be byte-identical to the
original chunk text — no translation, transliteration, or character
normalization is applied.

Property 15: For any set of PDF files where some have already been indexed,
running the pipeline without force-reprocess SHALL not re-process
already-indexed files, and vector store contents SHALL remain unchanged.

Property 16: For any set of PDF files where some fail during processing,
the pipeline SHALL successfully process all non-failing files, and
PipelineSummary counts SHALL satisfy:
files_processed + files_skipped + files_errored = total files.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.preprocessing.pipeline import PreprocessingPipeline
from sms_rag.shared.embedding import EmbeddingModel
from sms_rag.shared.models import (
    ConversationChunk,
    Message,
    ParsedConversation,
)
from sms_rag.shared.vector_store import VectorStoreInterface


# --- Generators ---

# Greek characters (uppercase and lowercase)
GREEK_CHARS = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίόύώ"
# Greeklish (Latin characters used phonetically for Greek)
GREEKLISH_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# English + digits + punctuation
ENGLISH_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"

# Combined multilingual alphabet for text generation
MULTILINGUAL_ALPHABET = GREEK_CHARS + GREEKLISH_CHARS + ENGLISH_CHARS + " "


@st.composite
def multilingual_text_generator(draw, min_size=1, max_size=200):
    """Generate text containing Greek, English, and/or Greeklish characters."""
    text = draw(
        st.text(
            alphabet=MULTILINGUAL_ALPHABET,
            min_size=min_size,
            max_size=max_size,
        )
    )
    return text


@st.composite
def file_list_generator(draw, min_files=1, max_files=10):
    """Generate a list of file descriptors with unique filenames.

    Each descriptor is a tuple of (filename, will_fail, already_indexed).
    Filenames are guaranteed unique by using index-based naming.
    """
    num_files = draw(st.integers(min_value=min_files, max_value=max_files))
    will_fail_flags = draw(
        st.lists(st.booleans(), min_size=num_files, max_size=num_files)
    )
    already_indexed_flags = draw(
        st.lists(st.booleans(), min_size=num_files, max_size=num_files)
    )

    files = []
    for i in range(num_files):
        filename = f"file_{i}.pdf"
        files.append((filename, will_fail_flags[i], already_indexed_flags[i]))
    return files


# --- Helper: Tracking Vector Store Mock ---


class TrackingVectorStore:
    """A mock vector store that tracks what documents were stored."""

    def __init__(self, indexed_filenames: set[str] | None = None):
        self.indexed_filenames = indexed_filenames or set()
        self.store_calls: list[dict[str, Any]] = []

    def has_document(self, source_filename: str) -> bool:
        return source_filename in self.indexed_filenames

    def store_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.store_calls.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )
        # Mark the files as indexed
        for meta in metadatas:
            if "source_filename" in meta:
                self.indexed_filenames.add(meta["source_filename"])

    def query_by_similarity(self, *args, **kwargs):
        return []

    def delete_by_ids(self, ids: list[str]) -> None:
        pass

    def get_all_stored_texts(self) -> list[str]:
        """Get all texts that were passed to store_embeddings."""
        all_texts = []
        for call in self.store_calls:
            all_texts.extend(call["documents"])
        return all_texts


# --- Helper: Build pipeline with controlled mocks ---


def _create_pipeline_with_mocks(
    data_dir: Path,
    file_descriptors: list[tuple[str, bool, bool]],
    chunk_texts: dict[str, list[str]] | None = None,
    force_reprocess: bool = False,
):
    """Create a pipeline with mock components based on file descriptors.

    Args:
        data_dir: Directory for PDF files.
        file_descriptors: List of (filename, will_fail, already_indexed) tuples.
        chunk_texts: Optional mapping of filename -> chunk texts to use.
        force_reprocess: Whether to force reprocessing.

    Returns:
        Tuple of (pipeline, tracking_vector_store).
    """
    # Determine which files are already indexed
    indexed_filenames = {
        fname for fname, _, already_indexed in file_descriptors if already_indexed
    }

    tracking_store = TrackingVectorStore(indexed_filenames=indexed_filenames)

    # Create dummy PDF files on disk
    for filename, _, _ in file_descriptors:
        pdf_path = data_dir / filename
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

    # Mock parser: returns success or error based on will_fail flag
    mock_parser = MagicMock(spec=PDFParser)

    def parser_side_effect(pdf_path):
        filename = pdf_path.name
        descriptor = next((d for d in file_descriptors if d[0] == filename), None)
        if descriptor is None or descriptor[1]:  # will_fail
            return ParsedConversation(
                participant_name=filename.replace(".pdf", ""),
                source_filename=filename,
                messages=[],
                errors=[f"Failed to parse {filename}"],
            )
        texts = (chunk_texts or {}).get(filename, [f"Message from {filename}"])
        messages = [Message(text=t) for t in texts]
        return ParsedConversation(
            participant_name=filename.replace(".pdf", ""),
            source_filename=filename,
            messages=messages,
            errors=[],
        )

    mock_parser.parse.side_effect = parser_side_effect

    # Mock chunker: produces chunks with the original text preserved
    mock_chunker = MagicMock(spec=ConversationChunker)

    def chunker_side_effect(conversation):
        if not conversation.messages:
            return []
        # Create a single chunk with all message texts joined by newline
        text = "\n".join(msg.text for msg in conversation.messages)
        chunk = ConversationChunk(
            chunk_id=f"chunk_{conversation.source_filename}_0",
            text=text,
            participant_name=conversation.participant_name,
            source_filename=conversation.source_filename,
            message_count=len(conversation.messages),
        )
        return [chunk]

    mock_chunker.chunk.side_effect = chunker_side_effect

    # Mock embedding model: returns dummy vectors
    mock_embedding_model = MagicMock(spec=EmbeddingModel)
    mock_embedding_model.embed_documents.side_effect = lambda texts: [
        [0.1] * 10 for _ in texts
    ]

    # Mock BM25 builder
    mock_bm25_builder = MagicMock(spec=BM25IndexBuilder)

    pipeline = PreprocessingPipeline(
        data_dir=data_dir,
        vector_store=tracking_store,
        embedding_model=mock_embedding_model,
        chunker=mock_chunker,
        parser=mock_parser,
        bm25_builder=mock_bm25_builder,
        force_reprocess=force_reprocess,
    )

    return pipeline, tracking_store


# --- Property 4: Text Preservation Through Pipeline ---


class TestTextPreservationThroughPipeline:
    """Property 4: Text Preservation Through Pipeline.

    # Feature: sms-rag, Property 4: Text Preservation Through Pipeline
    """

    @given(
        texts=st.lists(
            multilingual_text_generator(min_size=1, max_size=200),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_text_stored_is_byte_identical_to_original(self, texts):
        """For any conversation chunk containing Greek, English, or Greeklish
        text, the text stored in the vector store SHALL be byte-identical to the
        original chunk text.

        **Validates: Requirements 2.2, 2.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            filename = "TestFile.pdf"
            file_descriptors = [(filename, False, False)]

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_descriptors,
                chunk_texts={filename: texts},
            )

            pipeline.run()

            # Verify the text stored in vector store is byte-identical
            stored_texts = tracking_store.get_all_stored_texts()
            assert len(stored_texts) == 1

            # The expected stored text is the chunk text (messages joined by newline)
            expected_text = "\n".join(texts)
            actual_text = stored_texts[0]

            # Byte-identical comparison
            assert actual_text.encode("utf-8") == expected_text.encode("utf-8"), (
                f"Text not byte-identical.\n"
                f"Expected bytes: {expected_text.encode('utf-8')[:100]}\n"
                f"Actual bytes:   {actual_text.encode('utf-8')[:100]}"
            )

    @given(
        text=st.text(
            alphabet=GREEK_CHARS,
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_greek_text_preserved_without_transliteration(self, text):
        """Greek text is preserved without transliteration or normalization.

        **Validates: Requirements 2.2, 2.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            filename = "Greek.pdf"
            file_descriptors = [(filename, False, False)]

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_descriptors,
                chunk_texts={filename: [text]},
            )

            pipeline.run()

            stored_texts = tracking_store.get_all_stored_texts()
            assert len(stored_texts) == 1
            assert stored_texts[0] == text

    @given(
        text=st.text(
            alphabet=GREEKLISH_CHARS + " ",
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_greeklish_text_preserved_without_translation(self, text):
        """Greeklish text is preserved without translation to Greek.

        **Validates: Requirements 2.2, 2.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            filename = "Greeklish.pdf"
            file_descriptors = [(filename, False, False)]

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_descriptors,
                chunk_texts={filename: [text]},
            )

            pipeline.run()

            stored_texts = tracking_store.get_all_stored_texts()
            assert len(stored_texts) == 1
            assert stored_texts[0] == text


# --- Property 15: Pipeline Idempotence ---


class TestPipelineIdempotence:
    """Property 15: Pipeline Idempotence.

    # Feature: sms-rag, Property 15: Pipeline Idempotence
    """

    @given(
        file_list=file_list_generator(min_files=1, max_files=8),
    )
    @settings(max_examples=100)
    def test_already_indexed_files_not_reprocessed(self, file_list):
        """Running the pipeline without force-reprocess SHALL not re-process
        already-indexed files.

        **Validates: Requirements 8.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            pipeline.run()

            # Verify that no store_embeddings call included an already-indexed file
            already_indexed_files = {
                fname for fname, _, already_indexed in file_list if already_indexed
            }

            for call in tracking_store.store_calls:
                for meta in call["metadatas"]:
                    assert meta["source_filename"] not in already_indexed_files, (
                        f"Already-indexed file '{meta['source_filename']}' was "
                        f"re-processed (stored in vector store)"
                    )

    @given(
        file_list=file_list_generator(min_files=1, max_files=8),
    )
    @settings(max_examples=100)
    def test_vector_store_unchanged_for_indexed_files(self, file_list):
        """Vector store contents for already-indexed files SHALL remain unchanged.

        **Validates: Requirements 8.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            # Record initial state
            initial_store_calls_count = len(tracking_store.store_calls)
            assert initial_store_calls_count == 0  # Nothing stored yet

            pipeline.run()

            # Verify no store call references already-indexed files
            already_indexed_files = {
                fname for fname, _, already_indexed in file_list if already_indexed
            }

            for call in tracking_store.store_calls:
                for meta in call["metadatas"]:
                    source = meta["source_filename"]
                    assert source not in already_indexed_files, (
                        f"Already-indexed file '{source}' had embeddings stored, "
                        f"violating idempotence"
                    )

    @given(
        file_list=file_list_generator(min_files=2, max_files=8),
    )
    @settings(max_examples=100)
    def test_pipeline_skipped_count_matches_indexed_files(self, file_list):
        """The number of skipped files SHALL match the already-indexed files.

        **Validates: Requirements 8.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, _ = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            summary = pipeline.run()

            # Files that are already indexed should be skipped
            expected_skipped = sum(
                1 for _, _, already_indexed in file_list if already_indexed
            )

            assert summary.files_skipped == expected_skipped, (
                f"Expected {expected_skipped} skipped files, "
                f"got {summary.files_skipped}. "
                f"File list: {[(f, ai) for f, _, ai in file_list]}"
            )


# --- Property 16: Pipeline Resilience and Summary Accuracy ---


class TestPipelineResilienceAndSummaryAccuracy:
    """Property 16: Pipeline Resilience and Summary Accuracy.

    # Feature: sms-rag, Property 16: Pipeline Resilience and Summary Accuracy
    """

    @given(
        file_list=file_list_generator(min_files=1, max_files=10),
    )
    @settings(max_examples=100)
    def test_summary_counts_equal_total_files(self, file_list):
        """PipelineSummary counts SHALL satisfy:
        files_processed + files_skipped + files_errored = total files.

        **Validates: Requirements 8.4, 8.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, _ = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            summary = pipeline.run()

            total_files = len(file_list)
            actual_total = (
                summary.files_processed + summary.files_skipped + summary.files_errored
            )

            assert actual_total == total_files, (
                f"Summary counts don't add up: "
                f"processed={summary.files_processed} + "
                f"skipped={summary.files_skipped} + "
                f"errored={summary.files_errored} = {actual_total}, "
                f"expected {total_files}"
            )

    @given(
        file_list=file_list_generator(min_files=1, max_files=10),
    )
    @settings(max_examples=100)
    def test_non_failing_files_are_processed_successfully(self, file_list):
        """The pipeline SHALL successfully process all non-failing files.

        **Validates: Requirements 8.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            summary = pipeline.run()

            # Count expected outcomes:
            # - already_indexed -> skipped
            # - will_fail and not already_indexed -> errored
            # - not will_fail and not already_indexed -> processed
            expected_processed = sum(
                1
                for _, will_fail, already_indexed in file_list
                if not will_fail and not already_indexed
            )
            expected_skipped = sum(
                1 for _, _, already_indexed in file_list if already_indexed
            )
            expected_errored = sum(
                1
                for _, will_fail, already_indexed in file_list
                if will_fail and not already_indexed
            )

            assert summary.files_processed == expected_processed, (
                f"Expected {expected_processed} processed, "
                f"got {summary.files_processed}"
            )
            assert summary.files_skipped == expected_skipped, (
                f"Expected {expected_skipped} skipped, " f"got {summary.files_skipped}"
            )
            assert summary.files_errored == expected_errored, (
                f"Expected {expected_errored} errored, " f"got {summary.files_errored}"
            )

    @given(
        file_list=file_list_generator(min_files=2, max_files=10),
    )
    @settings(max_examples=100)
    def test_failing_files_do_not_prevent_others(self, file_list):
        """Failing files SHALL not prevent processing of non-failing files.

        **Validates: Requirements 8.4**
        """
        # Ensure at least one failing and one non-failing, non-indexed file
        has_failing = any(
            will_fail and not already_indexed
            for _, will_fail, already_indexed in file_list
        )
        has_processable = any(
            not will_fail and not already_indexed
            for _, will_fail, already_indexed in file_list
        )

        if not has_failing or not has_processable:
            # Skip this test case if the generated list doesn't have both
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, tracking_store = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            pipeline.run()

            # Non-failing, non-indexed files should be processed
            processable_files = {
                fname
                for fname, will_fail, already_indexed in file_list
                if not will_fail and not already_indexed
            }

            # Verify all processable files were stored
            stored_filenames = set()
            for call in tracking_store.store_calls:
                for meta in call["metadatas"]:
                    stored_filenames.add(meta["source_filename"])

            assert processable_files == stored_filenames, (
                f"Not all processable files were stored. "
                f"Expected: {processable_files}, Got: {stored_filenames}"
            )

    @given(
        file_list=file_list_generator(min_files=1, max_files=10),
    )
    @settings(max_examples=100)
    def test_error_count_matches_failing_files(self, file_list):
        """files_errored SHALL match the count of files that failed during
        processing.

        **Validates: Requirements 8.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            pipeline, _ = _create_pipeline_with_mocks(
                data_dir,
                file_list,
                force_reprocess=False,
            )

            summary = pipeline.run()

            # Only non-indexed files that will_fail should be errored
            expected_errored = sum(
                1
                for _, will_fail, already_indexed in file_list
                if will_fail and not already_indexed
            )

            assert summary.files_errored == expected_errored, (
                f"Expected {expected_errored} errored files, "
                f"got {summary.files_errored}"
            )
