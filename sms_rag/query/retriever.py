"""Hybrid retriever combining BM25 keyword search with semantic vector search.

Performs both retrieval methods independently, normalizes and merges scores
using a configurable weighting parameter, deduplicates by chunk_id, and
returns the top-k results sorted by final score descending.
"""

from pathlib import Path
from typing import Any

from sms_rag.query.bm25_search import BM25Search
from sms_rag.shared.embedding import EmbeddingModel
from sms_rag.shared.models import SearchResult
from sms_rag.shared.vector_store import VectorStoreInterface


class HybridRetriever:
    """Combines BM25 keyword search with semantic vector search.

    Both methods return their top-k candidates independently. Scores are
    normalized to [0.0, 1.0] within each method. The final merged score is:
        score = semantic_weight * vector_score + (1 - semantic_weight) * bm25_score

    Results are deduplicated by chunk_id (keeping the highest merged score)
    and returned sorted by final score descending.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedding_model: EmbeddingModel,
        bm25_index_path: Path,
        semantic_weight: float = 0.5,
        top_k: int = 5,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: Vector store interface for semantic search.
            embedding_model: Embedding model for query vectorization.
            bm25_index_path: Path to the pickled BM25 index file.
            semantic_weight: Weight for semantic scores in [0.0, 1.0].
                0.0 = keyword only, 1.0 = semantic only. Default: 0.5.
            top_k: Default number of top results to return.
        """
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._bm25 = BM25Search(bm25_index_path)
        self._semantic_weight = semantic_weight
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid retrieval combining BM25 and semantic search.

        Args:
            query: The search query string.
            metadata_filters: Optional metadata filters (participant_name,
                date range, etc.) to narrow the result set.
            top_k: Number of results to return. If None, uses the
                instance default.

        Returns:
            List of SearchResult objects sorted by merged score descending,
            with scores normalized to [0.0, 1.0]. Returns an empty list
            when no results are found from either method.
        """
        k = top_k if top_k is not None else self._top_k

        # Perform semantic vector search (supports metadata filters natively)
        query_embedding = self._embedding_model.embed_query(query)
        vector_results = self._vector_store.query_by_similarity(
            query_embedding=query_embedding,
            top_k=k,
            metadata_filters=metadata_filters,
        )

        # Perform BM25 keyword search (does not support metadata filters)
        bm25_results = self._bm25.search(query=query, top_k=k)

        # Post-filter BM25 results if metadata filters are provided
        if metadata_filters and bm25_results:
            bm25_results = self._apply_metadata_filters(bm25_results, metadata_filters)

        # Handle empty results from both methods
        if not vector_results and not bm25_results:
            return []

        # Merge scores using weighted fusion
        merged = self._merge_results(vector_results, bm25_results)

        # Sort by merged score descending and return top-k
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:k]

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge results from both methods using weighted score fusion.

        Deduplicates by chunk_id, keeping the highest merged score.

        Args:
            vector_results: Results from semantic vector search.
            bm25_results: Results from BM25 keyword search.

        Returns:
            Deduplicated list of SearchResult objects with merged scores.
        """
        w = self._semantic_weight

        # Build lookup maps by chunk_id
        vector_map: dict[str, SearchResult] = {r.chunk_id: r for r in vector_results}
        bm25_map: dict[str, SearchResult] = {r.chunk_id: r for r in bm25_results}

        # Collect all unique chunk_ids
        all_chunk_ids = set(vector_map.keys()) | set(bm25_map.keys())

        merged: dict[str, SearchResult] = {}
        for chunk_id in all_chunk_ids:
            vector_score = vector_map[chunk_id].score if chunk_id in vector_map else 0.0
            bm25_score = bm25_map[chunk_id].score if chunk_id in bm25_map else 0.0

            # Weighted fusion
            final_score = w * vector_score + (1 - w) * bm25_score

            # Use the result object from whichever method found it
            # (prefer vector result since it has metadata from the store)
            if chunk_id in vector_map:
                base_result = vector_map[chunk_id]
            else:
                base_result = bm25_map[chunk_id]

            merged_result = SearchResult(
                chunk_id=base_result.chunk_id,
                text=base_result.text,
                metadata=base_result.metadata,
                score=final_score,
            )

            # Deduplicate: keep highest merged score
            if chunk_id not in merged or final_score > merged[chunk_id].score:
                merged[chunk_id] = merged_result

        return list(merged.values())

    def _apply_metadata_filters(
        self,
        results: list[SearchResult],
        metadata_filters: dict[str, Any],
    ) -> list[SearchResult]:
        """Post-filter BM25 results by metadata.

        Since BM25Search doesn't support metadata filters natively,
        we filter its results after retrieval.

        Args:
            results: BM25 search results to filter.
            metadata_filters: Metadata key-value pairs to match.

        Returns:
            Filtered list of results matching all metadata conditions.
        """
        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, metadata_filters):
                filtered.append(result)
        return filtered

    @staticmethod
    def _matches_filters(
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Check if a result's metadata matches all filter conditions.

        Args:
            metadata: The result's metadata dictionary.
            filters: Required metadata key-value pairs.

        Returns:
            True if all filter conditions are satisfied.
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
