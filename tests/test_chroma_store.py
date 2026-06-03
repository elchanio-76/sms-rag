"""Unit tests for the ChromaDB vector store adapter."""

import tempfile
from pathlib import Path

import pytest

from sms_rag.shared.chroma_store import ChromaStore, VectorStoreError


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ChromaStore with a temporary directory."""
    return ChromaStore(persist_path=tmp_path / "chroma_test", timeout=30)


class TestChromaStoreInit:
    """Tests for ChromaStore initialization."""

    def test_creates_store_with_default_collection(self, tmp_path):
        store = ChromaStore(persist_path=tmp_path / "chroma")
        assert store._collection.name == "sms_conversations"

    def test_creates_store_with_string_path(self, tmp_path):
        store = ChromaStore(persist_path=str(tmp_path / "chroma"))
        assert store._collection is not None


class TestStoreEmbeddings:
    """Tests for store_embeddings method."""

    def test_store_and_retrieve_single_embedding(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        documents = ["Hello world"]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        # Verify by querying
        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5], top_k=1
        )
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Hello world"
        assert results[0].metadata["participant_name"] == "Alice"

    def test_store_multiple_embeddings(self, tmp_store):
        ids = ["chunk_1", "chunk_2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["First doc", "Second doc"]
        metadatas = [
            {"participant_name": "Alice", "source_filename": "alice.pdf"},
            {"participant_name": "Bob", "source_filename": "bob.pdf"},
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=5
        )
        assert len(results) == 2

    def test_store_metadata_with_list_fields(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test message"]
        metadatas = [
            {
                "participant_name": "Alice",
                "source_filename": "alice.pdf",
                "message_types": ["SMS", "iMessage"],
                "phone_numbers": ["+1234567890"],
            }
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=1
        )
        assert results[0].metadata["message_types"] == ["SMS", "iMessage"]
        assert results[0].metadata["phone_numbers"] == ["+1234567890"]

    def test_upsert_updates_existing_embedding(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Original text"]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        # Upsert with new document text
        tmp_store.store_embeddings(ids, embeddings, ["Updated text"], metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=1
        )
        assert results[0].text == "Updated text"


class TestQueryBySimilarity:
    """Tests for query_by_similarity method."""

    def test_returns_empty_list_when_no_documents(self, tmp_store):
        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=5
        )
        assert results == []

    def test_respects_top_k_limit(self, tmp_store):
        # Store 5 documents
        ids = [f"chunk_{i}" for i in range(5)]
        embeddings = [[float(i) * 0.1, 0.2, 0.3] for i in range(5)]
        documents = [f"Doc {i}" for i in range(5)]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}] * 5

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=3
        )
        assert len(results) <= 3

    def test_scores_are_in_valid_range(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test"]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=1
        )
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_results_sorted_by_score_descending(self, tmp_store):
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
        documents = ["A", "B", "C"]
        metadatas = [{"participant_name": "Alice", "source_filename": "a.pdf"}] * 3

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_filter_narrows_results(self, tmp_store):
        ids = ["chunk_1", "chunk_2"]
        embeddings = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        documents = ["From Alice", "From Bob"]
        metadatas = [
            {"participant_name": "Alice", "source_filename": "alice.pdf"},
            {"participant_name": "Bob", "source_filename": "bob.pdf"},
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            metadata_filters={"participant_name": "Alice"},
        )
        assert len(results) == 1
        assert results[0].metadata["participant_name"] == "Alice"

    def test_multiple_metadata_filters(self, tmp_store):
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        embeddings = [[0.1, 0.2, 0.3]] * 3
        documents = ["A", "B", "C"]
        metadatas = [
            {"participant_name": "Alice", "source_filename": "alice.pdf"},
            {"participant_name": "Alice", "source_filename": "other.pdf"},
            {"participant_name": "Bob", "source_filename": "bob.pdf"},
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            metadata_filters={
                "participant_name": "Alice",
                "source_filename": "alice.pdf",
            },
        )
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"


class TestDeleteByIds:
    """Tests for delete_by_ids method."""

    def test_delete_existing_embeddings(self, tmp_store):
        ids = ["chunk_1", "chunk_2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["A", "B"]
        metadatas = [
            {"participant_name": "Alice", "source_filename": "a.pdf"},
            {"participant_name": "Bob", "source_filename": "b.pdf"},
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)
        tmp_store.delete_by_ids(["chunk_1"])

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=5
        )
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_2"

    def test_delete_empty_list_is_noop(self, tmp_store):
        # Should not raise
        tmp_store.delete_by_ids([])


class TestHasDocument:
    """Tests for has_document method."""

    def test_returns_false_when_not_indexed(self, tmp_store):
        assert tmp_store.has_document("nonexistent.pdf") is False

    def test_returns_true_when_indexed(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test"]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)
        assert tmp_store.has_document("alice.pdf") is True

    def test_returns_false_for_different_filename(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test"]
        metadatas = [{"participant_name": "Alice", "source_filename": "alice.pdf"}]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)
        assert tmp_store.has_document("bob.pdf") is False


class TestMetadataSerialization:
    """Tests for metadata serialization/deserialization."""

    def test_none_values_are_excluded(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test"]
        metadatas = [
            {
                "participant_name": "Alice",
                "source_filename": "alice.pdf",
                "date_range_start": None,
                "date_range_end": None,
            }
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=1
        )
        # None fields should not be in the stored metadata
        assert "date_range_start" not in results[0].metadata
        assert "date_range_end" not in results[0].metadata

    def test_empty_list_fields_stored_as_json(self, tmp_store):
        ids = ["chunk_1"]
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["Test"]
        metadatas = [
            {
                "participant_name": "Alice",
                "source_filename": "alice.pdf",
                "message_types": [],
                "phone_numbers": [],
            }
        ]

        tmp_store.store_embeddings(ids, embeddings, documents, metadatas)

        results = tmp_store.query_by_similarity(
            query_embedding=[0.1, 0.2, 0.3], top_k=1
        )
        assert results[0].metadata["message_types"] == []
        assert results[0].metadata["phone_numbers"] == []
