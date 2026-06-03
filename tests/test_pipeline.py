"""Unit tests for the PreprocessingPipeline class."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.preprocessing.pipeline import PreprocessingPipeline
from sms_rag.shared.embedding import EmbeddingModel
from sms_rag.shared.models import (
    ConversationChunk,
    Message,
    ParsedConversation,
    PipelineSummary,
)
from sms_rag.shared.vector_store import VectorStoreInterface


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with fake PDF files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock(spec=VectorStoreInterface)
    store.has_document.return_value = False
    return store


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = MagicMock(spec=EmbeddingModel)
    # Return a simple embedding vector for each document
    model.embed_documents.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
    return model


@pytest.fixture
def mock_chunker():
    """Create a mock chunker."""
    chunker = MagicMock(spec=ConversationChunker)
    return chunker


@pytest.fixture
def mock_parser():
    """Create a mock PDF parser."""
    parser = MagicMock(spec=PDFParser)
    return parser


@pytest.fixture
def mock_bm25_builder():
    """Create a mock BM25 builder."""
    builder = MagicMock(spec=BM25IndexBuilder)
    return builder


def _create_pdf_file(data_dir: Path, name: str) -> Path:
    """Create a dummy PDF file in the data directory."""
    pdf_path = data_dir / name
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    return pdf_path


def _make_conversation(filename: str, n_messages: int = 3) -> ParsedConversation:
    """Create a ParsedConversation with n_messages messages."""
    return ParsedConversation(
        participant_name=filename.replace(".pdf", ""),
        source_filename=filename,
        messages=[
            Message(
                text=f"Message {i} from {filename}",
                timestamp=datetime(2024, 1, i + 1),
                message_type="SMS",
                phone_number="+1234567890",
            )
            for i in range(n_messages)
        ],
        errors=[],
    )


def _make_chunks(filename: str, n_chunks: int = 2) -> list[ConversationChunk]:
    """Create a list of ConversationChunk objects."""
    return [
        ConversationChunk(
            chunk_id=f"hash_{filename}_{i}",
            text=f"Chunk {i} text from {filename}",
            participant_name=filename.replace(".pdf", ""),
            source_filename=filename,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 10),
            message_types=["SMS"],
            phone_numbers=["+1234567890"],
            message_count=3,
        )
        for i in range(n_chunks)
    ]


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    def test_empty_data_directory(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline returns zero counts when no PDFs are found."""
        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_processed == 0
        assert summary.files_skipped == 0
        assert summary.files_errored == 0
        assert summary.chunks_created == 0
        assert summary.errors == []

    def test_processes_single_pdf(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline processes a single PDF file successfully."""
        _create_pdf_file(tmp_data_dir, "Alice.pdf")
        conversation = _make_conversation("Alice.pdf")
        chunks = _make_chunks("Alice.pdf")

        mock_parser.parse.return_value = conversation
        mock_chunker.chunk.return_value = chunks

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_processed == 1
        assert summary.files_skipped == 0
        assert summary.files_errored == 0
        assert summary.chunks_created == 2
        mock_vector_store.store_embeddings.assert_called_once()
        mock_bm25_builder.build.assert_called_once()

    def test_skips_already_indexed_files(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline skips files already in the vector store."""
        _create_pdf_file(tmp_data_dir, "Alice.pdf")
        mock_vector_store.has_document.return_value = True

        # Still need parser for BM25 re-indexing of skipped files
        conversation = _make_conversation("Alice.pdf")
        mock_parser.parse.return_value = conversation
        mock_chunker.chunk.return_value = _make_chunks("Alice.pdf")

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_processed == 0
        assert summary.files_skipped == 1
        assert summary.files_errored == 0
        assert summary.chunks_created == 0
        # store_embeddings should NOT be called for skipped files
        mock_vector_store.store_embeddings.assert_not_called()

    def test_force_reprocess_overrides_skip(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline reprocesses already-indexed files when force_reprocess=True."""
        _create_pdf_file(tmp_data_dir, "Alice.pdf")
        mock_vector_store.has_document.return_value = True

        conversation = _make_conversation("Alice.pdf")
        chunks = _make_chunks("Alice.pdf")
        mock_parser.parse.return_value = conversation
        mock_chunker.chunk.return_value = chunks

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
            force_reprocess=True,
        )

        summary = pipeline.run()

        assert summary.files_processed == 1
        assert summary.files_skipped == 0
        mock_vector_store.store_embeddings.assert_called_once()

    def test_continues_on_parse_error(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline continues processing when one file fails to parse."""
        _create_pdf_file(tmp_data_dir, "Bad.pdf")
        _create_pdf_file(tmp_data_dir, "Good.pdf")

        bad_conversation = ParsedConversation(
            participant_name="Bad",
            source_filename="Bad.pdf",
            messages=[],
            errors=["Corrupted PDF"],
        )
        good_conversation = _make_conversation("Good.pdf")
        good_chunks = _make_chunks("Good.pdf")

        mock_parser.parse.side_effect = [bad_conversation, good_conversation]
        mock_chunker.chunk.return_value = good_chunks

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_processed == 1
        assert summary.files_errored == 1
        assert len(summary.errors) == 1
        assert "Bad.pdf" in summary.errors[0]

    def test_continues_on_exception_during_processing(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Pipeline continues when an exception occurs during embedding/storage."""
        _create_pdf_file(tmp_data_dir, "Error.pdf")
        _create_pdf_file(tmp_data_dir, "Good.pdf")

        error_conversation = _make_conversation("Error.pdf")
        good_conversation = _make_conversation("Good.pdf")
        good_chunks = _make_chunks("Good.pdf")

        mock_parser.parse.side_effect = [error_conversation, good_conversation]
        # First call to chunk raises an exception, second succeeds
        mock_chunker.chunk.side_effect = [
            RuntimeError("Chunking failed"),
            good_chunks,
        ]

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_processed == 1
        assert summary.files_errored == 1
        assert "Error.pdf" in summary.errors[0]

    def test_summary_counts_add_up(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """files_processed + files_skipped + files_errored = total files."""
        _create_pdf_file(tmp_data_dir, "A.pdf")
        _create_pdf_file(tmp_data_dir, "B.pdf")
        _create_pdf_file(tmp_data_dir, "C.pdf")

        # A: skipped (already indexed)
        # B: errored (parse error)
        # C: processed successfully
        mock_vector_store.has_document.side_effect = [True, False, False]

        b_conv = ParsedConversation(
            participant_name="B",
            source_filename="B.pdf",
            messages=[],
            errors=["Parse failed"],
        )
        c_conv = _make_conversation("C.pdf")
        c_chunks = _make_chunks("C.pdf")

        mock_parser.parse.side_effect = [b_conv, c_conv]
        mock_chunker.chunk.return_value = c_chunks

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        total = summary.files_processed + summary.files_skipped + summary.files_errored
        assert total == 3
        assert summary.files_processed == 1
        assert summary.files_skipped == 1
        assert summary.files_errored == 1

    def test_bm25_index_built_with_all_chunks(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """BM25 index is built including chunks from skipped (previously indexed) files."""
        _create_pdf_file(tmp_data_dir, "Old.pdf")
        _create_pdf_file(tmp_data_dir, "New.pdf")

        # Old.pdf is already indexed, New.pdf is not
        mock_vector_store.has_document.side_effect = [True, False]

        old_conv = _make_conversation("Old.pdf", n_messages=5)
        new_conv = _make_conversation("New.pdf", n_messages=3)
        old_chunks = _make_chunks("Old.pdf", n_chunks=1)
        new_chunks = _make_chunks("New.pdf", n_chunks=1)

        # Parser is called for New.pdf during processing,
        # and for Old.pdf during BM25 rebuild
        mock_parser.parse.side_effect = [new_conv, old_conv]
        mock_chunker.chunk.side_effect = [new_chunks, old_chunks]

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        # BM25 builder should be called with chunks from both files
        mock_bm25_builder.build.assert_called_once()
        bm25_chunks = mock_bm25_builder.build.call_args[0][0]
        assert len(bm25_chunks) == 2  # 1 from new + 1 from old

    def test_prints_summary_to_stdout(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
        capsys,
    ):
        """Pipeline prints summary to stdout on completion."""
        _create_pdf_file(tmp_data_dir, "Test.pdf")
        conversation = _make_conversation("Test.pdf")
        chunks = _make_chunks("Test.pdf")

        mock_parser.parse.return_value = conversation
        mock_chunker.chunk.return_value = chunks

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        pipeline.run()

        captured = capsys.readouterr()
        assert "Preprocessing Pipeline Summary" in captured.out
        assert "Files processed:" in captured.out
        assert "Files skipped:" in captured.out
        assert "Files errored:" in captured.out
        assert "Chunks created:" in captured.out

    def test_empty_messages_counts_as_error(
        self,
        tmp_data_dir,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_parser,
        mock_bm25_builder,
    ):
        """Files that parse successfully but have no messages are counted as errors."""
        _create_pdf_file(tmp_data_dir, "Empty.pdf")

        empty_conv = ParsedConversation(
            participant_name="Empty",
            source_filename="Empty.pdf",
            messages=[],
            errors=[],
        )
        mock_parser.parse.return_value = empty_conv

        pipeline = PreprocessingPipeline(
            data_dir=tmp_data_dir,
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            chunker=mock_chunker,
            parser=mock_parser,
            bm25_builder=mock_bm25_builder,
        )

        summary = pipeline.run()

        assert summary.files_errored == 1
        assert summary.files_processed == 0
        assert "no messages extracted" in summary.errors[0]
