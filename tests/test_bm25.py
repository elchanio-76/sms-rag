"""Tests for BM25 index builder and search modules."""

import pickle
from pathlib import Path

import pytest

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.query.bm25_search import BM25Search
from sms_rag.shared.models import ConversationChunk


@pytest.fixture
def sample_chunks():
    """Create sample ConversationChunk objects for testing."""
    return [
        ConversationChunk(
            chunk_id="chunk_001",
            text="hello world this is a test message",
            participant_name="Alice",
            source_filename="Alice.pdf",
            message_count=3,
        ),
        ConversationChunk(
            chunk_id="chunk_002",
            text="goodbye world farewell message to all",
            participant_name="Alice",
            source_filename="Alice.pdf",
            message_count=2,
        ),
        ConversationChunk(
            chunk_id="chunk_003",
            text="hello again another greeting from me",
            participant_name="Bob",
            source_filename="Bob.pdf",
            message_count=4,
        ),
    ]


@pytest.fixture
def bm25_index_path(tmp_path):
    """Provide a temporary path for the BM25 index."""
    return tmp_path / "bm25_index.pkl"


class TestBM25IndexBuilder:
    """Tests for BM25IndexBuilder."""

    def test_build_creates_pickle_file(self, sample_chunks, bm25_index_path):
        """Build should create a pickle file at the specified path."""
        builder = BM25IndexBuilder(bm25_index_path)
        builder.build(sample_chunks)
        assert bm25_index_path.exists()

    def test_build_pickle_contains_required_keys(self, sample_chunks, bm25_index_path):
        """Pickle file should contain 'index', 'chunk_ids', and 'documents' keys."""
        builder = BM25IndexBuilder(bm25_index_path)
        builder.build(sample_chunks)

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)

        assert "index" in data
        assert "chunk_ids" in data
        assert "documents" in data

    def test_build_stores_correct_chunk_ids(self, sample_chunks, bm25_index_path):
        """Stored chunk IDs should match the input chunks."""
        builder = BM25IndexBuilder(bm25_index_path)
        builder.build(sample_chunks)

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)

        assert data["chunk_ids"] == ["chunk_001", "chunk_002", "chunk_003"]

    def test_build_stores_correct_documents(self, sample_chunks, bm25_index_path):
        """Stored documents should match the chunk texts."""
        builder = BM25IndexBuilder(bm25_index_path)
        builder.build(sample_chunks)

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)

        expected_docs = [chunk.text for chunk in sample_chunks]
        assert data["documents"] == expected_docs

    def test_build_creates_parent_directories(self, tmp_path, sample_chunks):
        """Build should create parent directories if they don't exist."""
        nested_path = tmp_path / "deep" / "nested" / "bm25_index.pkl"
        builder = BM25IndexBuilder(nested_path)
        builder.build(sample_chunks)
        assert nested_path.exists()

    def test_build_with_empty_chunks(self, bm25_index_path):
        """Build with empty list should still create a valid pickle file."""
        builder = BM25IndexBuilder(bm25_index_path)
        builder.build([])
        assert bm25_index_path.exists()

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)

        assert data["chunk_ids"] == []
        assert data["documents"] == []


class TestBM25Search:
    """Tests for BM25Search."""

    def _build_index(self, chunks, path):
        """Helper to build an index for testing."""
        builder = BM25IndexBuilder(path)
        builder.build(chunks)

    def test_search_returns_relevant_results(self, sample_chunks, bm25_index_path):
        """Search should return results matching the query terms."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello", top_k=5)
        assert len(results) > 0
        # Both chunk_001 and chunk_003 contain "hello"
        result_ids = [r.chunk_id for r in results]
        assert "chunk_001" in result_ids
        assert "chunk_003" in result_ids

    def test_search_returns_search_result_objects(self, sample_chunks, bm25_index_path):
        """Search should return SearchResult instances."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello")
        for result in results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "text")
            assert hasattr(result, "metadata")
            assert hasattr(result, "score")

    def test_search_scores_normalized(self, sample_chunks, bm25_index_path):
        """All result scores should be in [0.0, 1.0]."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello world")
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_search_top_result_has_score_one(self, sample_chunks, bm25_index_path):
        """The top result should have a normalized score of 1.0."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello world")
        assert len(results) > 0
        assert results[0].score == 1.0

    def test_search_respects_top_k(self, sample_chunks, bm25_index_path):
        """Search should return at most top_k results."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello", top_k=1)
        assert len(results) <= 1

    def test_search_results_sorted_descending(self, sample_chunks, bm25_index_path):
        """Results should be sorted by score in descending order."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello world message")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_no_match_returns_empty(self, sample_chunks, bm25_index_path):
        """Search with no matching terms should return empty list."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("xyznonexistent")
        assert results == []

    def test_search_metadata_is_empty_dict(self, sample_chunks, bm25_index_path):
        """BM25 search results should have empty metadata dicts."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("hello")
        for result in results:
            assert result.metadata == {}

    def test_search_result_text_matches_original(self, sample_chunks, bm25_index_path):
        """Result text should match the original chunk text."""
        self._build_index(sample_chunks, bm25_index_path)
        searcher = BM25Search(bm25_index_path)

        results = searcher.search("goodbye farewell")
        # chunk_002 should be the top match
        assert any(r.text == "goodbye world farewell message to all" for r in results)

    def test_file_not_found_raises(self, tmp_path):
        """Loading a non-existent index file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BM25Search(tmp_path / "nonexistent.pkl")
