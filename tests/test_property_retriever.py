"""Property-based tests for hybrid retrieval (Properties 8, 9, 10).

# Feature: sms-rag, Property 8: Hybrid Search Invokes Both Methods
# Feature: sms-rag, Property 9: Retrieval Scoring and Ranking Invariants
# Feature: sms-rag, Property 10: Metadata Filter Enforcement

Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from rank_bm25 import BM25Okapi

from sms_rag.query.retriever import HybridRetriever
from sms_rag.shared.models import SearchResult
from sms_rag.shared.vector_store import VectorStoreInterface


# --- Generators ---


@st.composite
def search_result_generator(draw, chunk_id=None, metadata=None):
    """Generate a random SearchResult object with scores in [0.0, 1.0]."""
    cid = chunk_id or draw(
        st.text(
            min_size=3,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=("L", "N"), blacklist_characters="\x00"
            ),
        ).map(lambda s: f"chunk_{s}")
    )
    text = draw(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"), blacklist_characters="\x00"
            ),
        )
    )
    score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    meta = metadata if metadata is not None else draw(metadata_for_filter_generator())

    return SearchResult(chunk_id=cid, text=text, metadata=meta, score=score)


@st.composite
def metadata_for_filter_generator(draw):
    """Generate metadata dicts suitable for filter testing."""
    participant = draw(st.sampled_from(["Alice", "Bob", "Charlie", "Diana", "Eve"]))
    source = draw(st.sampled_from(["file1.pdf", "file2.pdf", "file3.pdf"]))
    return {
        "participant_name": participant,
        "source_filename": source,
    }


@st.composite
def query_generator(draw):
    """Generate random query strings with at least one word."""
    words = draw(
        st.lists(
            st.sampled_from(
                [
                    "hello",
                    "world",
                    "test",
                    "search",
                    "message",
                    "conversation",
                    "phone",
                    "date",
                    "time",
                    "name",
                    "text",
                    "query",
                    "find",
                    "chat",
                    "call",
                ]
            ),
            min_size=1,
            max_size=5,
        )
    )
    return " ".join(words)


@st.composite
def semantic_weight_generator(draw):
    """Generate semantic weights in [0.0, 1.0]."""
    return draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )


@st.composite
def top_k_generator(draw):
    """Generate top_k values between 1 and 20."""
    return draw(st.integers(min_value=1, max_value=20))


@st.composite
def documents_generator(draw):
    """Generate a list of documents (at least 3) for BM25 indexing."""
    word_pool = [
        "hello",
        "world",
        "test",
        "search",
        "message",
        "conversation",
        "phone",
        "date",
        "time",
        "name",
        "text",
        "query",
        "find",
        "chat",
        "call",
        "good",
        "morning",
        "evening",
        "night",
        "day",
    ]
    num_docs = draw(st.integers(min_value=3, max_value=15))
    docs = []
    for _ in range(num_docs):
        words = draw(st.lists(st.sampled_from(word_pool), min_size=2, max_size=8))
        docs.append(" ".join(words))
    return docs


# --- Helpers ---


def _create_bm25_index(
    tmp_dir: str, documents: list[str], chunk_ids: list[str]
) -> Path:
    """Create a BM25 index pickle file from documents."""
    tokenized = [doc.split() for doc in documents]
    index = BM25Okapi(tokenized) if tokenized else None

    index_path = Path(tmp_dir) / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(
            {"index": index, "chunk_ids": chunk_ids, "documents": documents},
            f,
        )
    return index_path


class TrackingVectorStore(VectorStoreInterface):
    """Fake vector store that tracks calls and returns configurable results."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results or []
        self.query_called = False
        self.last_metadata_filters = None

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
        self.query_called = True
        self.last_metadata_filters = metadata_filters
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


class TrackingEmbeddingModel:
    """Fake embedding model that returns a dummy vector."""

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 10

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


# --- Property 8: Hybrid Search Invokes Both Methods ---


class TestHybridSearchInvokesBothMethods:
    """Property 8: Hybrid Search Invokes Both Methods.

    # Feature: sms-rag, Property 8: Hybrid Search Invokes Both Methods

    For any query string submitted to the hybrid retriever, both the BM25
    keyword search and the semantic vector search SHALL be invoked, and the
    final result set SHALL contain candidates sourced from both methods
    (when both produce results).
    """

    @given(
        query=query_generator(),
        semantic_weight=semantic_weight_generator(),
        top_k=top_k_generator(),
    )
    @settings(max_examples=25)
    def test_both_methods_invoked(self, query, semantic_weight, top_k):
        """Both BM25 and semantic search SHALL be invoked for any query.

        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create BM25 index with documents containing query terms
            documents = [
                "hello world message",
                "test search query",
                "conversation phone date",
                "find chat call text",
                "name time message search",
            ]
            chunk_ids = [f"chunk_{i}" for i in range(len(documents))]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            # Create vector results that overlap with some BM25 results
            vector_results = [
                SearchResult(
                    chunk_id=f"vec_chunk_{i}",
                    text=f"vector result {i}",
                    metadata={
                        "participant_name": "Alice",
                        "source_filename": "file1.pdf",
                    },
                    score=0.9 - i * 0.1,
                )
                for i in range(min(top_k, 3))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=top_k,
            )

            # Patch BM25Search.search to track if it was called
            with patch.object(
                retriever._bm25, "search", wraps=retriever._bm25.search
            ) as mock_bm25_search:
                results = retriever.retrieve(query)

                # Both methods should be invoked
                assert (
                    vector_store.query_called
                ), "Semantic vector search was NOT invoked"
                assert mock_bm25_search.called, "BM25 keyword search was NOT invoked"

    @given(
        query=query_generator(),
        semantic_weight=semantic_weight_generator(),
    )
    @settings(max_examples=25)
    def test_results_contain_candidates_from_both_methods(self, query, semantic_weight):
        """When both methods produce results, the final set SHALL contain
        candidates from both.

        **Validates: Requirements 5.1**
        """
        # Use a weight that values both methods (avoid 0.0 and 1.0 extremes
        # where one method's contribution would be zeroed out)
        assume(0.01 < semantic_weight < 0.99)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create BM25 index with documents that will match query words
            query_words = query.split()
            # Ensure at least one doc contains a query word for BM25 match
            documents = [
                " ".join(query_words + ["extra", "words"]),
                "unrelated document content here",
                "another unrelated text entry",
            ]
            chunk_ids = ["bm25_chunk_0", "bm25_chunk_1", "bm25_chunk_2"]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            # Vector store returns distinct results (different chunk_ids)
            vector_results = [
                SearchResult(
                    chunk_id="vec_chunk_0",
                    text="semantic match result",
                    metadata={
                        "participant_name": "Alice",
                        "source_filename": "file1.pdf",
                    },
                    score=0.9,
                ),
                SearchResult(
                    chunk_id="vec_chunk_1",
                    text="another semantic result",
                    metadata={
                        "participant_name": "Bob",
                        "source_filename": "file2.pdf",
                    },
                    score=0.7,
                ),
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=10,
            )

            results = retriever.retrieve(query)

            # Both methods produce results, so merged set should contain both
            result_ids = {r.chunk_id for r in results}
            vector_ids = {r.chunk_id for r in vector_results}
            bm25_ids = set(chunk_ids)

            has_vector_candidates = bool(result_ids & vector_ids)
            has_bm25_candidates = bool(result_ids & bm25_ids)

            assert has_vector_candidates, (
                f"No vector candidates in results. "
                f"Result IDs: {result_ids}, Vector IDs: {vector_ids}"
            )
            assert has_bm25_candidates, (
                f"No BM25 candidates in results. "
                f"Result IDs: {result_ids}, BM25 IDs: {bm25_ids}"
            )


# --- Property 9: Retrieval Scoring and Ranking Invariants ---


class TestRetrievalScoringAndRankingInvariants:
    """Property 9: Retrieval Scoring and Ranking Invariants.

    # Feature: sms-rag, Property 9: Retrieval Scoring and Ranking Invariants

    For any set of search results with a semantic weight W in [0.0, 1.0],
    the merged score for each result SHALL equal W * normalized_vector_score
    + (1 - W) * normalized_bm25_score, all scores SHALL be in [0.0, 1.0],
    the result count SHALL not exceed top-k, and results SHALL be sorted
    descending.
    """

    @given(
        query=query_generator(),
        semantic_weight=semantic_weight_generator(),
        top_k=top_k_generator(),
    )
    @settings(max_examples=25)
    def test_all_scores_in_valid_range(self, query, semantic_weight, top_k):
        """All scores SHALL be in the range [0.0, 1.0].

        **Validates: Requirements 5.2, 5.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents = [
                "hello world message test",
                "search query conversation phone",
                "date time name text find",
                "chat call good morning evening",
                "night day hello search message",
            ]
            chunk_ids = [f"chunk_{i}" for i in range(len(documents))]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            vector_results = [
                SearchResult(
                    chunk_id=f"chunk_{i}",
                    text=documents[i],
                    metadata={"participant_name": "Alice", "source_filename": "f.pdf"},
                    score=0.9 - i * 0.15,
                )
                for i in range(min(top_k, 4))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=top_k,
            )

            results = retriever.retrieve(query)

            for r in results:
                assert 0.0 <= r.score <= 1.0, (
                    f"Score {r.score} out of range [0.0, 1.0] "
                    f"for chunk_id={r.chunk_id}"
                )

    @given(
        query=query_generator(),
        semantic_weight=semantic_weight_generator(),
        top_k=top_k_generator(),
    )
    @settings(max_examples=25)
    def test_result_count_not_exceeds_top_k(self, query, semantic_weight, top_k):
        """Result count SHALL not exceed the configured top-k.

        **Validates: Requirements 5.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create many documents to potentially exceed top_k
            documents = [
                f"document number {i} with words search test" for i in range(20)
            ]
            chunk_ids = [f"chunk_{i}" for i in range(20)]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            vector_results = [
                SearchResult(
                    chunk_id=f"vec_{i}",
                    text=f"vector doc {i}",
                    metadata={"participant_name": "Alice", "source_filename": "f.pdf"},
                    score=0.95 - i * 0.05,
                )
                for i in range(10)
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=top_k,
            )

            results = retriever.retrieve(query)
            assert (
                len(results) <= top_k
            ), f"Got {len(results)} results, exceeds top_k={top_k}"

    @given(
        query=query_generator(),
        semantic_weight=semantic_weight_generator(),
        top_k=top_k_generator(),
    )
    @settings(max_examples=25)
    def test_results_sorted_descending(self, query, semantic_weight, top_k):
        """Results SHALL be sorted in descending score order.

        **Validates: Requirements 5.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents = [
                "hello world test",
                "search message conversation",
                "phone date time name",
                "find chat call text",
                "good morning evening night",
            ]
            chunk_ids = [f"chunk_{i}" for i in range(len(documents))]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            vector_results = [
                SearchResult(
                    chunk_id=f"chunk_{i}",
                    text=documents[i],
                    metadata={"participant_name": "Alice", "source_filename": "f.pdf"},
                    score=0.9 - i * 0.1,
                )
                for i in range(len(documents))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=top_k,
            )

            results = retriever.retrieve(query)

            if len(results) > 1:
                scores = [r.score for r in results]
                for i in range(len(scores) - 1):
                    assert scores[i] >= scores[i + 1], (
                        f"Results not sorted descending: "
                        f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
                    )

    @given(
        semantic_weight=semantic_weight_generator(),
    )
    @settings(max_examples=25)
    def test_merged_score_formula(self, semantic_weight):
        """Merged score SHALL equal W * vector_score + (1-W) * bm25_score.

        **Validates: Requirements 5.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use multiple documents so BM25 IDF is non-zero for "hello world"
            # (BM25 IDF = 0 when term appears in all docs with only 1 doc)
            documents = ["hello world", "foo bar baz", "unrelated content"]
            chunk_ids = ["shared_chunk", "other_1", "other_2"]
            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            # Both methods return the same chunk with known scores
            vector_score = 0.8
            vector_results = [
                SearchResult(
                    chunk_id="shared_chunk",
                    text="hello world",
                    metadata={"participant_name": "Alice", "source_filename": "f.pdf"},
                    score=vector_score,
                ),
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=semantic_weight,
                top_k=5,
            )

            results = retriever.retrieve("hello world")

            # BM25: "hello world" is the best match among multiple docs,
            # normalized to 1.0 (only it contains both query terms)
            bm25_score = 1.0
            expected_score = (
                semantic_weight * vector_score + (1 - semantic_weight) * bm25_score
            )

            # Find the shared chunk
            shared_results = [r for r in results if r.chunk_id == "shared_chunk"]
            assert (
                len(shared_results) == 1
            ), f"Expected shared_chunk in results, got {[r.chunk_id for r in results]}"

            actual_score = shared_results[0].score
            assert abs(actual_score - expected_score) < 1e-6, (
                f"Score mismatch: expected {expected_score} "
                f"(W={semantic_weight} * vec={vector_score} + "
                f"(1-W)={1-semantic_weight} * bm25={bm25_score}), "
                f"got {actual_score}"
            )


# --- Property 10: Metadata Filter Enforcement ---


class TestMetadataFilterEnforcement:
    """Property 10: Metadata Filter Enforcement.

    # Feature: sms-rag, Property 10: Metadata Filter Enforcement

    For any query with metadata filters applied (participant name, date range),
    all returned SearchResult objects SHALL satisfy the filter conditions —
    no result with non-matching metadata SHALL appear in the output.
    """

    @given(
        query=query_generator(),
        filter_participant=st.sampled_from(["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        top_k=top_k_generator(),
    )
    @settings(max_examples=25)
    def test_participant_name_filter_enforced(self, query, filter_participant, top_k):
        """All returned results SHALL satisfy participant_name filter condition.

        **Validates: Requirements 5.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create BM25 index where documents have associated metadata
            # Simulate a real scenario with mixed participants
            all_participants = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
            documents = []
            chunk_ids = []
            doc_metadata = []

            for i, participant in enumerate(all_participants):
                doc_text = f"{query.split()[0] if query.split() else 'hello'} message from {participant} number {i}"
                documents.append(doc_text)
                chunk_ids.append(f"chunk_{participant}_{i}")
                doc_metadata.append(
                    {
                        "participant_name": participant,
                        "source_filename": f"{participant}.pdf",
                    }
                )

            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            # Vector store has results from multiple participants
            vector_results = [
                SearchResult(
                    chunk_id=chunk_ids[i],
                    text=documents[i],
                    metadata=doc_metadata[i],
                    score=0.9 - i * 0.1,
                )
                for i in range(len(documents))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=0.5,
                top_k=top_k,
            )

            metadata_filters = {"participant_name": filter_participant}
            results = retriever.retrieve(query, metadata_filters=metadata_filters)

            # All results must satisfy the filter
            for r in results:
                if r.metadata.get("participant_name"):
                    assert r.metadata["participant_name"] == filter_participant, (
                        f"Result {r.chunk_id} has participant_name="
                        f"'{r.metadata['participant_name']}', "
                        f"expected '{filter_participant}'"
                    )

    @given(
        query=query_generator(),
        filter_source=st.sampled_from(["file1.pdf", "file2.pdf", "file3.pdf"]),
    )
    @settings(max_examples=25)
    def test_source_filename_filter_enforced(self, query, filter_source):
        """All returned results SHALL satisfy source_filename filter condition.

        **Validates: Requirements 5.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            sources = ["file1.pdf", "file2.pdf", "file3.pdf"]
            documents = []
            chunk_ids = []
            doc_metadata = []

            for i, source in enumerate(sources):
                doc_text = f"content from {source} with {query.split()[0] if query.split() else 'data'} info {i}"
                documents.append(doc_text)
                chunk_ids.append(f"chunk_{source}_{i}")
                doc_metadata.append(
                    {
                        "participant_name": f"Person{i}",
                        "source_filename": source,
                    }
                )

            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            vector_results = [
                SearchResult(
                    chunk_id=chunk_ids[i],
                    text=documents[i],
                    metadata=doc_metadata[i],
                    score=0.85 - i * 0.1,
                )
                for i in range(len(documents))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=0.5,
                top_k=10,
            )

            metadata_filters = {"source_filename": filter_source}
            results = retriever.retrieve(query, metadata_filters=metadata_filters)

            # All results must satisfy the source_filename filter
            for r in results:
                if r.metadata.get("source_filename"):
                    assert r.metadata["source_filename"] == filter_source, (
                        f"Result {r.chunk_id} has source_filename="
                        f"'{r.metadata['source_filename']}', "
                        f"expected '{filter_source}'"
                    )

    @given(
        query=query_generator(),
        filter_participant=st.sampled_from(["Alice", "Bob", "Charlie"]),
        filter_source=st.sampled_from(["file1.pdf", "file2.pdf", "file3.pdf"]),
    )
    @settings(max_examples=25)
    def test_combined_filters_enforced(self, query, filter_participant, filter_source):
        """All results SHALL satisfy ALL filter conditions simultaneously.

        **Validates: Requirements 5.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            participants = ["Alice", "Bob", "Charlie"]
            sources = ["file1.pdf", "file2.pdf", "file3.pdf"]

            documents = []
            chunk_ids = []
            doc_metadata = []

            idx = 0
            for participant in participants:
                for source in sources:
                    doc_text = f"{query.split()[0] if query.split() else 'hello'} from {participant} in {source} idx {idx}"
                    documents.append(doc_text)
                    chunk_ids.append(f"chunk_{idx}")
                    doc_metadata.append(
                        {
                            "participant_name": participant,
                            "source_filename": source,
                        }
                    )
                    idx += 1

            index_path = _create_bm25_index(tmp_dir, documents, chunk_ids)

            vector_results = [
                SearchResult(
                    chunk_id=chunk_ids[i],
                    text=documents[i],
                    metadata=doc_metadata[i],
                    score=0.95 - i * 0.05,
                )
                for i in range(len(documents))
            ]

            vector_store = TrackingVectorStore(results=vector_results)
            embedding_model = TrackingEmbeddingModel()

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                bm25_index_path=index_path,
                semantic_weight=0.5,
                top_k=10,
            )

            metadata_filters = {
                "participant_name": filter_participant,
                "source_filename": filter_source,
            }
            results = retriever.retrieve(query, metadata_filters=metadata_filters)

            # All results must satisfy BOTH filter conditions
            for r in results:
                if r.metadata.get("participant_name"):
                    assert r.metadata["participant_name"] == filter_participant, (
                        f"Result {r.chunk_id} has participant_name="
                        f"'{r.metadata['participant_name']}', "
                        f"expected '{filter_participant}'"
                    )
                if r.metadata.get("source_filename"):
                    assert r.metadata["source_filename"] == filter_source, (
                        f"Result {r.chunk_id} has source_filename="
                        f"'{r.metadata['source_filename']}', "
                        f"expected '{filter_source}'"
                    )
