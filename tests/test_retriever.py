"""Unit tests for the HybridRetriever class."""

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sms_rag.query.retriever import HybridRetriever
from sms_rag.shared.models import SearchResult
from sms_rag.shared.vector_store import VectorStoreInterface


def _create_bm25_index(
    tmp_path: Path, documents: list[str], chunk_ids: list[str]
) -> Path:
    """Helper to create a BM25 index pickle file for testing."""
    from rank_bm25 import BM25Okapi

    tokenized = [doc.split() for doc in documents]
    index = BM25Okapi(tokenized) if tokenized else None

    index_path = tmp_path / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(
            {"index": index, "chunk_ids": chunk_ids, "documents": documents},
            f,
        )
    return index_path


def _create_empty_bm25_index(tmp_path: Path) -> Path:
    """Helper to create an empty BM25 index."""
    index_path = tmp_path / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"index": None, "chunk_ids": [], "documents": []}, f)
    return index_path


class FakeVectorStore(VectorStoreInterface):
    """Fake vector store that returns preconfigured results."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results or []

    def store_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        pass

    def query_by_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        results = self._results
        if metadata_filters:
            results = [
                r
                for r in results
                if all(r.metadata.get(k) == v for k, v in metadata_filters.items())
            ]
        return results[:top_k]

    def delete_by_ids(self, ids: list[str]) -> None:
        pass

    def has_document(self, source_filename: str) -> bool:
        return False


class FakeEmbeddingModel:
    """Fake embedding model that returns a dummy vector."""

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 10

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_returns_empty_when_no_results(self, tmp_path):
        """Should return empty list when both methods return no results."""
        index_path = _create_empty_bm25_index(tmp_path)
        vector_store = FakeVectorStore(results=[])
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
        )

        results = retriever.retrieve("hello")
        assert results == []

    def test_merges_vector_and_bm25_results(self, tmp_path):
        """Should merge results from both methods with weighted scores."""
        documents = ["hello world", "foo bar baz", "another document"]
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        # Vector store returns chunk_1 and chunk_2
        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=0.9
            ),
            SearchResult(
                chunk_id="chunk_2", text="foo bar baz", metadata={}, score=0.6
            ),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=5,
        )

        results = retriever.retrieve("hello world")

        # Should have results from both methods
        assert len(results) > 0
        # All scores should be in [0.0, 1.0]
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_results_sorted_by_score_descending(self, tmp_path):
        """Should return results sorted by final score descending."""
        documents = ["hello world", "foo bar", "hello foo"]
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=0.9
            ),
            SearchResult(chunk_id="chunk_2", text="foo bar", metadata={}, score=0.3),
            SearchResult(chunk_id="chunk_3", text="hello foo", metadata={}, score=0.5),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=5,
        )

        results = retriever.retrieve("hello")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, tmp_path):
        """Should return at most top_k results."""
        documents = [f"doc {i}" for i in range(20)]
        chunk_ids = [f"chunk_{i}" for i in range(20)]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"doc {i}",
                metadata={},
                score=0.9 - i * 0.04,
            )
            for i in range(10)
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=3,
        )

        results = retriever.retrieve("doc")
        assert len(results) <= 3

    def test_top_k_override_in_retrieve(self, tmp_path):
        """Should allow overriding top_k per query."""
        documents = [f"doc {i}" for i in range(10)]
        chunk_ids = [f"chunk_{i}" for i in range(10)]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"doc {i}",
                metadata={},
                score=0.9 - i * 0.05,
            )
            for i in range(10)
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=10,
        )

        results = retriever.retrieve("doc", top_k=2)
        assert len(results) <= 2

    def test_deduplicates_by_chunk_id(self, tmp_path):
        """Should deduplicate results by chunk_id, keeping highest merged score."""
        documents = ["hello world", "foo bar"]
        chunk_ids = ["chunk_1", "chunk_2"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        # Both methods return chunk_1
        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=0.8
            ),
            SearchResult(chunk_id="chunk_2", text="foo bar", metadata={}, score=0.4),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=5,
        )

        results = retriever.retrieve("hello world")
        # chunk_1 should appear only once
        chunk_1_results = [r for r in results if r.chunk_id == "chunk_1"]
        assert len(chunk_1_results) == 1

    def test_metadata_filters_applied_to_vector_search(self, tmp_path):
        """Should apply metadata filters to vector store search."""
        documents = ["hello world", "foo bar"]
        chunk_ids = ["chunk_1", "chunk_2"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id="chunk_1",
                text="hello world",
                metadata={"participant_name": "Alice"},
                score=0.9,
            ),
            SearchResult(
                chunk_id="chunk_2",
                text="foo bar",
                metadata={"participant_name": "Bob"},
                score=0.7,
            ),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=5,
        )

        results = retriever.retrieve(
            "hello", metadata_filters={"participant_name": "Alice"}
        )
        # Vector store should filter to only Alice's results
        # BM25 results without metadata should also be post-filtered
        for r in results:
            # Results from vector search will have the metadata
            if r.metadata.get("participant_name"):
                assert r.metadata["participant_name"] == "Alice"

    def test_semantic_weight_only_vector(self, tmp_path):
        """With semantic_weight=1.0, should use only vector scores."""
        documents = ["hello world"]
        chunk_ids = ["chunk_1"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=0.8
            ),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=1.0,
            top_k=5,
        )

        results = retriever.retrieve("hello world")
        # With weight=1.0, the chunk_1 score should be 1.0 * 0.8 + 0.0 * bm25_score
        # Since bm25 also returns chunk_1, its bm25_score won't contribute
        chunk_1 = next(r for r in results if r.chunk_id == "chunk_1")
        # Score should be 1.0 * 0.8 + 0.0 * bm25 = 0.8
        assert abs(chunk_1.score - 0.8) < 0.01

    def test_semantic_weight_only_bm25(self, tmp_path):
        """With semantic_weight=0.0, should use only BM25 scores."""
        # Need multiple docs so BM25 IDF is non-zero for query terms
        documents = ["hello world", "foo bar baz", "something else"]
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=0.8
            ),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.0,
            top_k=5,
        )

        results = retriever.retrieve("hello world")
        # With weight=0.0, score = 0.0 * vector + 1.0 * bm25_score
        chunk_1 = next(r for r in results if r.chunk_id == "chunk_1")
        # BM25 best match "hello world" should get normalized score of 1.0
        assert abs(chunk_1.score - 1.0) < 0.01

    def test_scores_in_valid_range(self, tmp_path):
        """All merged scores should be in [0.0, 1.0]."""
        documents = ["hello world", "foo bar baz", "testing one two three"]
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        index_path = _create_bm25_index(tmp_path, documents, chunk_ids)

        vector_results = [
            SearchResult(
                chunk_id="chunk_1", text="hello world", metadata={}, score=1.0
            ),
            SearchResult(
                chunk_id="chunk_2", text="foo bar baz", metadata={}, score=0.5
            ),
            SearchResult(
                chunk_id="chunk_3", text="testing one two three", metadata={}, score=0.2
            ),
        ]
        vector_store = FakeVectorStore(results=vector_results)
        embedding_model = FakeEmbeddingModel()

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index_path=index_path,
            semantic_weight=0.5,
            top_k=5,
        )

        results = retriever.retrieve("hello")
        for r in results:
            assert 0.0 <= r.score <= 1.0
